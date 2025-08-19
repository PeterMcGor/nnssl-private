from loguru import logger
import numpy as np 
from nnssl.training.loss.mse_loss import LossMaskMSELoss, MAEMSELoss
import torch
from torch import nn
import torch.nn.functional as F



class NaiveProjector(nn.Module):
    """Original direct projection approach"""
    def __init__(self, bottleneck_dim, subject_dim):
        super().__init__()
        bottleneck_flat_dim = np.prod(np.array(bottleneck_dim))    
        self.projector = nn.Sequential(
            nn.Linear(bottleneck_flat_dim, subject_dim * 2),  # Intermediate layer
            nn.LayerNorm(subject_dim * 2),
            nn.ReLU(),
            nn.Linear(subject_dim * 2, subject_dim),     # Target: subject_dim
            nn.LayerNorm(subject_dim)
        )
    
    def forward(self, latent, batch_size):
        # Standard flattening
        bottleneck_flat = latent.view(batch_size, -1)  # [batch, 40000]
        return self.projector(bottleneck_flat)


class ProgressiveProjector(nn.Module):
    """Progressive reduction approach"""
    def __init__(self, bottleneck_dim, subject_dim):
        super().__init__()
        bottleneck_flat_dim = np.prod(np.array(bottleneck_dim))
        self.projector = nn.Sequential(
            nn.Linear(bottleneck_flat_dim, 2048),     
            nn.LayerNorm(2048),
            nn.ReLU(),
            #nn.Dropout(0.1), Regularitation probably not needed here
            
            nn.Linear(2048, 512),                 
            nn.LayerNorm(512),
            nn.ReLU(),
            #nn.Dropout(0.1),
            
            nn.Linear(512, subject_dim),          
            nn.LayerNorm(subject_dim)
        )
    
    def forward(self, latent, batch_size):
        # Standard flattening
        bottleneck_flat = latent.view(batch_size, -1)  
        return self.projector(bottleneck_flat)



class PooledProgressiveProjector(nn.Module):
    """Spatial pooling + progressive reduction with channel portion selection"""
    
    def __init__(self, bottleneck_dim, subject_dim, channel_portion=1.0):
        """
        Args:
            bottleneck_dim: Original bottleneck dimensions as shape (e.g., [320, 5, 5, 5])
            subject_dim: Target embedding dimension
            channel_portion: Portion of channels to use (0.0 to 1.0)
                - 0.0: Use 1 channel (minimum to avoid errors)
                - 1.0: Use all channels
                - 0.5: Use half the channels (first half)
        """
        super().__init__()
        
        # For compatibility with bottleneck_flat_dim = np.prod(np.array(bottleneck_dim))
        if isinstance(bottleneck_dim, (list, tuple)):
            self.bottleneck_shape = bottleneck_dim
            total_channels = bottleneck_dim[0]  # 320
            self.spatial_dims = bottleneck_dim[1:]  # [5, 5, 5]
        else:
            # If it's already flattened dimension, assume default shape
            raise ValueError("bottleneck_dim should be the shape [channels, d, h, w], not flattened dimension")
        
        # Calculate number of channels to use based on portion
        self.channel_portion = max(0.0, min(1.0, channel_portion))  # Clamp between 0 and 1
        n_channels_to_use = max(1, int(total_channels * self.channel_portion))  # At least 1 channel
        
        # Select first n_channels_to_use channels
        self.selected_channels = list(range(n_channels_to_use))
        
        # Calculate dimensions after pooling and channel selection
        pooled_spatial = 2 * 2 * 2  # After adaptive_avg_pool3d to (2,2,2)
        bottleneck_flat_dim = n_channels_to_use * pooled_spatial
        
        print(f"Using {n_channels_to_use}/{total_channels} channels ({self.channel_portion:.1%})")
        print(f"Flattened dimension after pooling: {bottleneck_flat_dim}")
        
        self.projector = nn.Sequential(
            nn.Linear(bottleneck_flat_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            
            nn.Linear(256, subject_dim),
            nn.LayerNorm(subject_dim)
        )
    
    def forward(self, latent, batch_size):
        # Select portion of channels BEFORE pooling: [batch, 320, 5, 5, 5] -> [batch, n_selected, 5, 5, 5]
        if len(self.selected_channels) < latent.shape[1]:
            latent_selected = latent[:, self.selected_channels, :, :, :]
        else:
            latent_selected = latent
        
        # Apply spatial pooling to selected channels: [batch, n_selected, 5, 5, 5] -> [batch, n_selected, 2, 2, 2]
        pooled = F.adaptive_avg_pool3d(latent_selected, (2, 2, 2))
        # Flatten: [batch, n_selected * 2 * 2 * 2]
        bottleneck_flat = pooled.view(batch_size, -1)
        return self.projector(bottleneck_flat)


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
                 similarity_weight=1.0, variance_weight=0.1,
                 image_projector=None, channels_proportion_at_embedding = 1.0):  # Pass module object here
        super().__init__()
        self.similarity_weight = similarity_weight
        self.variance_weight = variance_weight
        
        # Use provided projector or default to ProgressiveProjector
        if image_projector is None:
            self.image_to_subject_projector = PooledProgressiveProjector(bottleneck_dim, subject_dim, channel_portion=channels_proportion_at_embedding)
        else:
            self.image_to_subject_projector = image_projector
        
        # NO learnable projection for subjects!
        
    def normalize_subjects(self, subject_features):
        """Simple, deterministic normalization - preserves structure"""
        # Option 1: Just L2 normalization
        return F.normalize(subject_features, dim=1, eps=1e-8)
        
       
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
        # CRITICAL FIX: Row-wise normalization (per case, independent of batch)
        #row_means = subject_features.mean(dim=1, keepdim=True)
        #row_stds = subject_features.std(dim=1, keepdim=True, unbiased=False)
        
        # Avoid division by zero: if std is too small, don't normalize that row
        #safe_std = torch.where(row_stds < 1e-6, torch.ones_like(row_stds), row_stds)
        #subject_features = (subject_features - row_means) / safe_std
        
        # Clip extreme values after normalization
        #subject_features = torch.clamp(subject_features, min=-10.0, max=10.0)
            
        return subject_features, subject_ids
    
    def compute_similarity_loss(self, image_emb, subject_emb):
        """
        Compute direct cosine similarity between corresponding pairs.
        High similarity = low loss.
        """
        # Add safety checks before normalization
        if torch.isnan(image_emb).any() or torch.isnan(subject_emb).any():
            print("WARNING: NaN in embeddings before similarity")
            return torch.tensor(0.0, device=image_emb.device), torch.zeros(image_emb.shape[0], device=image_emb.device)
        
        # Row-wise L2 normalization with safety for zero vectors
        def safe_normalize(x):
            norms = torch.norm(x, dim=1, keepdim=True)
            safe_norms = torch.where(norms < 1e-8, torch.ones_like(norms), norms)
            return x / safe_norms
        
        # Normalize embeddings (essential for avoiding trivial solutions)
        image_emb = safe_normalize(image_emb)#F.normalize(image_emb, dim=1, eps=1e-8)
        subject_emb = safe_normalize(subject_emb)#F.normalize(subject_emb, dim=1, eps=1e-8)
        
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
        #subject_std = torch.std(subject_emb, dim=0, unbiased=False).mean()
        
        # Encourage std to be at least 0.1 (adjustable target)
        target_std = 0.1
        
        # Only penalize when std is below target (using ReLU)
        image_penalty = torch.clamp(target_std - image_std, min=0, max=100)
        #subject_penalty = torch.relu(target_std - subject_std)
        
        return image_penalty #+ subject_penalty
    
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


        # Add safety checks for each row
        if torch.isnan(subject_features).any():
            print("WARNING: NaN detected in subject_features")
            subject_features = torch.nan_to_num(subject_features, nan=0.0)
        
        # Check for rows with extreme values and handle them individually
        #row_maxes = subject_features.abs().max(dim=1, keepdim=True)[0]
        #extreme_rows = row_maxes > 1e6
        #if extreme_rows.any():
        #    print(f"WARNING: {extreme_rows.sum()} rows with extreme values")
            # Clip extreme rows individually
        #    subject_features = torch.where(
        #        extreme_rows.expand_as(subject_features),
        #        torch.clamp(subject_features, min=-1e6, max=1e6),
        #        subject_features
        #    )
        
        latent = latent[0]
        batch_size = latent.shape[0]
        
        # Project images to subject space
        image_projected = self.image_to_subject_projector(latent, batch_size)


        # Check for NaN after projection
        if torch.isnan(image_projected).any():
            print("WARNING: NaN detected in image_projected")
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        
        # Subjects: just normalize (preserve original meaning!)
        subject_normalized = self.normalize_subjects(subject_features)
        
        # Direct comparison in subject space
        similarity_loss, cosine_sims = self.compute_similarity_loss(image_projected, subject_normalized)
        
        # Only regularize the learnable part (image projections)
        #variance_loss = self.compute_variance_regularization(image_projected, image_projected)
        
        total_loss = self.similarity_weight * similarity_loss #+ self.variance_weight * variance_loss)
        #print(f"Similarity Loss: {similarity_loss.item()}, Variance Loss: {variance_loss.item()}")
        #stats = self.compute_embedding_stats(image_projected, subject_normalized)
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
            bottleneck_dim: Dimensions of the latent bottleneck
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
            logger.info(f"Recon Loss: {recon_loss:.6f}, Sim Loss: {sim_loss:.6f}, Total: {total_loss:.6f} with {len(self.similarity_loss.image_to_subject_projector.selected_channels)} bottleneck channels")
        
        return total_loss

class ReconstructionAndSimilarityLossPortion05(ReconstructionAndSimilarityLoss):
    """
    Inherits from ReconstructionAndSimilarityLoss but uses only 50% of channels 
    for the similarity loss computation.
    """
    
    def __init__(self,
                 # Similarity loss parameters
                 bottleneck_dim, 
                 subject_dim,
                 similarity_kwargs=None,
                 # Loss weights
                 weight_reconstruction=1.0, 
                 weight_similarity=0.1):
        """
        Initialize with same parameters as parent, but automatically set 
        channels_proportion_at_embedding to 0.5
        
        Args:
            bottleneck_dim: Dimensions of the latent bottleneck 
            subject_dim: Dimension of subject features
            similarity_kwargs: Dict of kwargs for SubjectImageSimilarityLoss
            weight_reconstruction: Weight for reconstruction loss
            weight_similarity: Weight for similarity loss
        """
        
        # Default similarity loss parameters
        if similarity_kwargs is None:
            similarity_kwargs = {
                'similarity_weight': 1.0,
                'variance_weight': 0.1
            }
        
        # Add the 50% channel portion parameter
        similarity_kwargs['channels_proportion_at_embedding'] = 0.5
        
        # Call parent constructor with modified similarity_kwargs
        super().__init__(
            bottleneck_dim=bottleneck_dim,
            subject_dim=subject_dim,
            similarity_kwargs=similarity_kwargs,
            weight_reconstruction=weight_reconstruction,
            weight_similarity=weight_similarity
        )

