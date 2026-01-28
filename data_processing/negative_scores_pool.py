def collect_negative_scores(model, train_dataset, bool_multiclass=True, num_classes=10, device='cuda'):
    if bool_multiclass:
        negative_scores_pools = {i: [] for i in range(num_classes)}
    else:
        negative_scores_pools = []
    model.eval()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            features = model.get_features(images)
            scores = model.fc2(features)
            if bool_multiclass:
                for class_idx in range(NUM_CLASSES):
                    neg_mask = labels != class_idx
                    if neg_mask.sum() > 0:
                        class_scores = scores[neg_mask, class_idx]
                        negative_scores_pools[class_idx].append(class_scores.cpu())
            else:
                neg_mask = labels != 1
                if neg_mask.sum() > 0:
                        class_scores = scores[neg_mask]
                        negative_scores_pools.append(class_scores.cpu())
    if bool_multiclass:
        for class_idx in range(NUM_CLASSES):
            if negative_scores_pools[class_idx]:
                negative_scores_pools[class_idx] = torch.cat(negative_scores_pools[class_idx]).numpy()
                print(f"Class {class_idx}: collected {len(negative_scores_pools[class_idx])} negative scores")
            else:
                negative_scores_pools[class_idx] = np.array([])
                print(f"Class {class_idx}: no negative scores collected")
    return negative_scores_pools