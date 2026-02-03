import torch

# Charge l'ancien checkpoint
ckpt = torch.load("/home/ibendali/checkpoints/last.pt", map_location="cpu")

# Affiche toutes les clés pour voir s'il y a un historique caché
print("Clés disponibles :", ckpt.keys())

# Si tu vois une clé 'loss_history' ou 'train_losses', c'est gagné !