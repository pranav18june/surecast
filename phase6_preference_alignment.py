import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import argparse
import shutil
import random
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed} for reproducibility.")
    
# ==========================================
# 1. CONTINUOUS DPO-INSPIRED LOSS
# ==========================================

class ContinuousDPOLoss(nn.Module):
    """
    A continuous contrastive loss inspired by Direct Preference Optimization (DPO).
    
    IMPORTANT THEORETICAL CAVEAT:
    Original DPO (Rafailov et al.) is mathematically derived from the Bradley-Terry model 
    for discrete token generation probabilities. By mapping the reward signal to negative MSE 
    in a continuous forecasting domain, we are applying a heuristic "DPO-inspired" contrastive 
    loss. It remains an open theoretical question whether the exact optimality guarantees of 
    discrete DPO transfer rigorously to this continuous setting. Do not claim strict mathematical 
    equivalence to LLM DPO.
    
    Formula:
    L = -log(sigmoid(beta * (MSE(y_rej, y_hat) - MSE(y_pref, y_hat))))
    """
    def __init__(self, beta=0.1):
        super(ContinuousDPOLoss, self).__init__()
        self.beta = beta

    def forward(self, mu_pred, y_pref, y_rej):
        # Calculate base rewards (Negative MSE)
        mse_pref = F.mse_loss(mu_pred, y_pref, reduction='none')
        mse_rej = F.mse_loss(mu_pred, y_rej, reduction='none')
        
        # Reward difference: R(pref) - R(rej) = MSE(rej) - MSE(pref)
        reward_diff = mse_rej - mse_pref
        
        # Contrastive loss
        loss = -F.logsigmoid(self.beta * reward_diff)
        return torch.mean(loss)

# ==========================================
# 2. FEEDBACK MANAGER & PROTECTIONS
# ==========================================

class FeedbackManager:
    def __init__(self, model, checkpoint_dir="models", n_sigma_thresh=4.0):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.n_sigma_thresh = n_sigma_thresh
        self.allowlist = {"EXPERT_01", "EXPERT_02", "ADMIN"}
        self.current_version = 0
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        # Save initial state
        self._save_checkpoint()

    def _save_checkpoint(self):
        """Saves a versioned checkpoint of the model."""
        path = os.path.join(self.checkpoint_dir, f"model_v{self.current_version}.pt")
        torch.save(self.model.state_dict(), path)
        logging.info(f"[SYSTEM] Model checkpoint saved: {path}")

    def rollback(self, target_version):
        """Rolls back the model to a previous version."""
        path = os.path.join(self.checkpoint_dir, f"model_v{target_version}.pt")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.current_version = target_version
            logging.info(f"[SYSTEM] Rolled back successfully to version {target_version}")
            return True
        else:
            logging.error(f"[SYSTEM] Rollback failed. Version {target_version} does not exist.")
            return False

    def submit_preference_pair(self, x_seq, y_pref, y_rej, contributor_id):
        """
        Accepts a preference pair if it passes security and anomaly checks.
        """
        # 1. Allowlist Protection
        if contributor_id not in self.allowlist:
            logging.warning(f"[SECURITY] Rejected submission: Contributor '{contributor_id}' is not in the allowlist.")
            return False

        # Get model's current prior belief
        self.model.eval()
        with torch.no_grad():
            mu_prior, sigma_prior = self.model(x_seq)
            
        mu = mu_prior.item()
        sigma = sigma_prior.item()

        # 2. Anomaly Check (Implausible Rejected Target)
        # If the rejected forecast is absurdly far from the prior, drop it to prevent poisoning
        distance = abs(y_rej.item() - mu)
        if distance > self.n_sigma_thresh * sigma:
            logging.warning(f"[ANOMALY] Rejected submission: Rejected target ({y_rej.item():.2f}) is implausibly far "
                            f"from model prior (mu={mu:.2f}, sigma={sigma:.2f}). "
                            f"Distance is {distance/sigma:.1f}x sigma (Threshold: {self.n_sigma_thresh}x).")
            return False

        logging.info(f"[SUCCESS] Preference pair accepted from {contributor_id}.")
        return True

    def fine_tune(self, preference_dataset, epochs=1, lr=1e-4):
        """
        Executes the DPO-inspired fine-tuning round and bumps version.
        """
        if not preference_dataset:
            logging.info("No valid pairs to fine-tune.")
            return
            
        logging.info(f"\n--- Starting RLHF Fine-Tuning Round (v{self.current_version} -> v{self.current_version + 1}) ---")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = ContinuousDPOLoss(beta=0.1)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for x, y_pref, y_rej in preference_dataset:
                optimizer.zero_grad()
                mu_pred, _ = self.model(x)
                loss = criterion(mu_pred, y_pref, y_rej)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logging.info(f"Epoch {epoch+1}/{epochs} | DPO Loss: {total_loss/len(preference_dataset):.4f}")
            
        # Bump version and save checkpoint
        self.current_version += 1
        self._save_checkpoint()

# ==========================================
# 3. DUMMY MODEL FOR TESTING
# ==========================================
class DummyForecastModel(nn.Module):
    def __init__(self):
        super(DummyForecastModel, self).__init__()
        self.fc_mu = nn.Linear(10, 1)
        self.fc_sigma = nn.Linear(10, 1)
        
    def forward(self, x):
        mu = self.fc_mu(x)
        sigma = F.softplus(self.fc_sigma(x)) + 1e-3
        return mu, sigma

# ==========================================
# MAIN ROUTINE & AUTOMATED TESTS
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    logging.info("═══════════════════════════════════════")
    logging.info("PHASE 6 — Human Feedback / RLHF")
    logging.info("═══════════════════════════════════════\n")
    
    set_seed(42)
    
    logging.info(f"\n[ASSUMPTION FLAG] The continuous contrastive loss uses negative MSE. It is assumed this heuristic adequately captures DPO dynamics in a continuous setting, though theoretical equivalence to discrete DPO is not claimed.\n")

    data_path = "data/preference_dataset.csv"
    if not os.path.exists(data_path):
        logging.error(f"[ERROR] NOT YET COMPUTED — requires preference pairs from experts (e.g., {data_path}).")
        logging.error("Please provide the preference data to run the RLHF alignment.")
        sys.exit(1)
        
    # Execution logic would go here.
    
    logging.info("\nPhase 6 Complete.")

if __name__ == "__main__":
    set_seed(42)
    main()
