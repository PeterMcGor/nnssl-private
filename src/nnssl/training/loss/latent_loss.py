from loguru import logger
from nnssl.training.loss.mse_loss import LossMaskMSELoss, MAEMSELoss
import torch
from torch import nn
import torch.nn.functional as F


class BottleNeckContrastiveLoss(nn.Module):
    def __init__(self, feat_weight: float = 0.1, in_dim=1, out_dim=1):
        super().__init__()
        self.feat_weight = feat_weight
        self.proj_latent = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, batch, output, mask, latent):        
        #print(len(latent)) # why is this a list??
        #print(latent[0].shape)
        return torch.mean(latent[0]) + torch.mean(output**2) 


class SubjectImageSimilarityLoss(nn.Module):
    def __init__(self, bottleneck_dim, subject_dim, 
                 similarity_weight=1.0, variance_weight=0.1):
        super().__init__()
        self.similarity_weight = similarity_weight
        self.variance_weight = variance_weight
        
        # Project images to SUBJECT dimension (learnable)
        self.image_to_subject_projector = nn.Sequential(
            nn.Linear(bottleneck_dim, subject_dim * 2),  # Intermediate layer
            nn.LayerNorm(subject_dim * 2),
            nn.ReLU(),
            nn.Linear(subject_dim * 2, subject_dim),     # Target: subject_dim
            nn.LayerNorm(subject_dim)
        )
        
        # NO learnable projection for subjects!
        
    def normalize_subjects(self, subject_features):
        """Simple, deterministic normalization - preserves structure"""
        # Option 1: Just L2 normalization
        return F.normalize(subject_features, dim=1, eps=1e-8)
        
        # Option 2: Standardization (if you prefer)
        # return (subject_features - subject_features.mean(dim=0)) / (subject_features.std(dim=0) + 1e-8)
        
    def extract_subject_data(self, batch, device='cuda'):
        """Extract subject features and IDs from batch dictionary."""

        subject_features = []
        
        for i in range(len(batch['properties'])):
            case_values = list(batch['properties'][i]['subject_features'].values())
            subject_features.append(case_values)
        
        subject_ids = batch.get('subject_ids', None) # TODO
        
        if subject_features is None:
            raise KeyError("'subject_features' not found in batch")

        subject_features = torch.tensor(subject_features, dtype=torch.float32)
        subject_features = subject_features.to(device)
        
        # CRITICAL FIX: Normalize the huge values (1.6M, 200K, etc.) 
        # Your subject features are causing numerical overflow -> NaN
        #subject_features = (subject_features - subject_features.mean(dim=0, keepdim=True)) / (subject_features.std(dim=0, keepdim=True) + 1e-8)
            
        return subject_features, subject_ids
    
    def compute_similarity_loss(self, image_emb, subject_emb):
        """
        Compute direct cosine similarity between corresponding pairs.
        High similarity = low loss.
        """
        # Normalize embeddings (essential for avoiding trivial solutions)
        image_emb = F.normalize(image_emb, dim=1, eps=1e-8)
        subject_emb = F.normalize(subject_emb, dim=1, eps=1e-8)
        
        # Compute cosine similarity for each pair
        cosine_similarities = F.cosine_similarity(image_emb, subject_emb, dim=1)
        
        # We want high similarity, so minimize negative similarity
        similarity_loss = -cosine_similarities.mean()
        
        return similarity_loss, cosine_similarities
    
    def compute_variance_regularization(self, image_emb, subject_emb):
        """
        Encourage embeddings to have sufficient spread using standard deviation.
        Most numerically stable approach.
        """
        eps = 1e-6
        
        # Compute standard deviation across batch for each dimension, then average
        image_std = torch.std(image_emb, dim=0, unbiased=False).mean()
        subject_std = torch.std(subject_emb, dim=0, unbiased=False).mean()
        
        # Encourage std to be at least 0.1 (adjustable target)
        target_std = 0.1
        
        # Only penalize when std is below target (using ReLU)
        image_penalty = torch.relu(target_std - image_std)
        subject_penalty = torch.relu(target_std - subject_std)
        
        return image_penalty + subject_penalty
    
    def compute_covariance_regularization(self, image_emb, subject_emb):
        """
        Decorrelate features - normalized version for numerical stability.
        """
        def off_diagonal_covariance(x):
            if x.size(0) <= 1:  # Skip if batch too small
                return torch.tensor(0.0, device=x.device)
            
            # Center the data
            x_centered = x - x.mean(dim=0, keepdim=True)
            
            # Compute covariance matrix
            cov = torch.mm(x_centered.T, x_centered) / (x.size(0) - 1)
            
            # Get off-diagonal elements using mask
            mask = ~torch.eye(cov.size(0), dtype=torch.bool, device=cov.device)
            off_diag_elements = cov[mask]
            
            # Return mean of squared off-diagonal elements (normalized!)
            return torch.mean(off_diag_elements**2)
        
        image_cov_loss = off_diagonal_covariance(image_emb)
        subject_cov_loss = off_diagonal_covariance(subject_emb)
    
        return image_cov_loss + subject_cov_loss
    
    def compute_embedding_stats(self, image_emb, subject_emb):
        """Check if embeddings are actually diverse"""
        # Compute pairwise distances within each modality
        image_distances = torch.pdist(image_emb).mean()
        subject_distances = torch.pdist(subject_emb).mean()
        
        # Compute standard deviation across the batch
        image_std = torch.std(image_emb, dim=0).mean()
        subject_std = torch.std(subject_emb, dim=0).mean()
        
        return {
            'image_pairwise_dist': image_distances.item(),
            'subject_pairwise_dist': subject_distances.item(), 
            'image_std': image_std.item(),
            'subject_std': subject_std.item()
        }

    
    def forward(self, batch, output, mask, latent):
        subject_features, _ = self.extract_subject_data(batch, output.device)
        
        latent = latent[0]
        batch_size = latent.shape[0]
        
        # Project images to subject space
        bottleneck_flat = latent.view(batch_size, -1)
        image_projected = self.image_to_subject_projector(bottleneck_flat)
        
        # Subjects: just normalize (preserve original meaning!)
        subject_normalized = self.normalize_subjects(subject_features)
        
        # Direct comparison in subject space
        similarity_loss, cosine_sims = self.compute_similarity_loss(image_projected, subject_normalized)
        
        # Only regularize the learnable part (image projections)
        variance_loss = self.compute_variance_regularization(image_projected, image_projected)
        
        total_loss = (self.similarity_weight * similarity_loss +
                     self.variance_weight * variance_loss)
        #print(f"Similarity Loss: {similarity_loss.item()}, Variance Loss: {variance_loss.item()}")
        stats = self.compute_embedding_stats(image_projected, subject_normalized)
        #print(f"Embedding Stats: {stats}")
        
        return total_loss


class AlternativeVICRegLoss(nn.Module):
    """
    Alternative approach using VICReg-style loss (Variance-Invariance-Covariance).
    Very robust against collapse.
    """
    def __init__(self, bottleneck_dim, subject_dim, projection_dim=256, 
                 sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight 
        self.cov_weight = cov_weight
        
        self.image_projector = nn.Sequential(
            nn.Linear(bottleneck_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.subject_projector = nn.Sequential(
            nn.Linear(subject_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def extract_subject_data(self, batch):
        subject_features = batch.get('subject_features', None)
        subject_ids = batch.get('subject_ids', None)
        
        if subject_features is None or subject_ids is None:
            raise KeyError("Missing subject data in batch")
            
        return subject_features, subject_ids
    
    def invariance_loss(self, z1, z2):
        """Minimize distance between corresponding pairs."""
        return F.mse_loss(z1, z2)
    
    def variance_loss(self, z):
        """Maintain variance above threshold."""
        std_z = torch.sqrt(z.var(dim=0) + 1e-04)
        return torch.mean(F.relu(1 - std_z))
    
    def covariance_loss(self, z):
        """Decorrelate features."""
        batch_size, dim = z.shape
        z = z - z.mean(dim=0)
        cov_z = (z.T @ z) / (batch_size - 1)
        
        # Off-diagonal elements should be close to 0
        off_diag_cov = cov_z.flatten()[:-1].view(dim-1, dim+1)[:, 1:].flatten()
        return off_diag_cov.pow(2).sum() / dim
    
    def forward(self, batch, output, mask, latent):
        subject_features, subject_ids = self.extract_subject_data(batch)
        
        batch_size = latent.size(0)
        bottleneck_flat = latent.view(batch_size, -1)
        
        # Project both modalities
        z_image = self.image_projector(bottleneck_flat)
        z_subject = self.subject_projector(subject_features)
        
        # VICReg loss components
        inv_loss = self.invariance_loss(z_image, z_subject)
        var_loss = self.variance_loss(z_image) + self.variance_loss(z_subject)
        cov_loss = self.covariance_loss(z_image) + self.covariance_loss(z_subject)
        
        total_loss = (self.sim_weight * inv_loss + 
                     self.var_weight * var_loss + 
                     self.cov_weight * cov_loss)
        
        # Store for monitoring
        self.last_inv_loss = inv_loss.item()
        self.last_var_loss = var_loss.item()
        self.last_cov_loss = cov_loss.item()
        
        return total_loss


class ReconstructionAndSimilarityLoss(nn.Module):
    def __init__(self, 
                 # Similarity loss parameters
                 bottleneck_dim, 
                 subject_dim,
                 similarity_kwargs=None,
                 # Loss weights
                 weight_reconstruction=1.0, 
                 weight_similarity=0.1):
        """
        Compound loss combining reconstruction (MSE) and subject-image similarity.
        
        Args:
            bottleneck_dim: Dimension of the latent bottleneck
            subject_dim: Dimension of subject features
            similarity_kwargs: Dict of kwargs for SubjectImageSimilarityLoss
            weight_reconstruction: Weight for reconstruction loss
            weight_similarity: Weight for similarity loss
        """
        super().__init__()
        
        self.weight_reconstruction = weight_reconstruction
        self.weight_similarity = weight_similarity
        
        # Initialize the component losses
        self.reconstruction_loss = MAEMSELoss()
        
        # Default similarity loss parameters
        if similarity_kwargs is None:
            similarity_kwargs = {
                'similarity_weight': 1.0,
                'variance_weight': 0.1
            }
        
        self.similarity_loss = SubjectImageSimilarityLoss(
            bottleneck_dim=bottleneck_dim,
            subject_dim=subject_dim,
            **similarity_kwargs
        )
    
    def forward(self, batch, model_output, target, mask, latent):
        """
        Forward pass computing both reconstruction and similarity losses.
        
        Args:
            batch: Batch dictionary containing subject features
            model_output: Model's reconstruction output
            target: Ground truth target
            loss_mask: Mask for reconstruction loss calculation
            latent: Latent representations from the model
            
        Returns:
            Combined weighted loss
        """
        
        # Compute reconstruction loss (original task)
        recon_loss = self.reconstruction_loss.forward(
            model_output=model_output,
            target=target, 
            mask=mask
        ) if self.weight_reconstruction != 0 else 0
        
        # Compute similarity loss (new contrastive task)
        sim_loss = self.similarity_loss.forward(
            batch=batch,
            output=model_output,  # Pass model_output as 'output'
            mask=mask,
            latent=latent
        ) if self.weight_similarity != 0 else 0
        
        # Combine losses with weights
        total_loss = (self.weight_reconstruction * recon_loss + 
                     self.weight_similarity * sim_loss)
        
        # Optional: Print component losses for monitoring
        if self.weight_reconstruction != 0 and self.weight_similarity != 0:
            logger.info(f"Recon Loss: {recon_loss:.6f}, Sim Loss: {sim_loss:.6f}, Total: {total_loss:.6f}")
        
        return total_loss

