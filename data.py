import torch
import torch.utils.data as data
import json
import pandas as pd
import h5py
import re


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and video features
    """

    def __init__(self, data_path, data_split, pretrained_path, vocab):
        self.vocab = vocab

        with open(data_path + data_split + ".json", "r") as f:
            self.info = pd.Series(json.load(f)).apply(pd.Series)
        self.info = self.info[self.info["description"].str.split(" ").str.len() >= 2].reset_index(drop=True)

        self.info['features_path'] = self.info['video'].apply(
            lambda x: pretrained_path + "fc7_subsample5_fps25_" + x + ".h5")

        self.length = len(self.info)

    @staticmethod
    def load_pretrained_features(path):
        features = None
        with h5py.File(path, "r") as hf:
            features = torch.Tensor(hf["features"])
        return features

    def __getitem__(self, idx):
        # handle the image redundancy
        image = self.load_pretrained_features(self.info["features_path"][idx])
        caption = re.sub(r"[^a-z\- ]", "", self.info["description"][idx]\
                         .replace("'s", "")).split(" ")
        y_true = torch.tensor(self.info["times"][idx][:4])

        target = []
        for token in caption:
            try:
                target.append(int(self.vocab.stoi[token]))
            except KeyError:
                pass
        if len(target) <= 1:
            target = []
            target.append(int(self.vocab.stoi["start"]))
            target.append(int(self.vocab.stoi["end"]))

        return image, torch.Tensor(target), y_true

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, y_true = zip(*data)

    img_lens = [len(img) for img in images]
    max_len = max(img_lens)
    padded_images = torch.zeros(len(images), max_len, images[0].shape[-1]).float()

    for i, img in enumerate(images):
        end = img_lens[i]
        padded_images[i, :end] = img[:end]

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    cap_lens = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(cap_lens)).long()
    for i, cap in enumerate(captions):
        end = cap_lens[i]
        targets[i, :end] = cap[:end]

    return padded_images, targets, img_lens, cap_lens, torch.stack(y_true)


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, opt.pretrained_path, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(vocab, batch_size, workers, opt):
    train_loader = get_precomp_loader(opt.data_path, 'train_data', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(opt.data_path, 'val_data', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(vocab, batch_size, workers, opt):
    test_loader = get_precomp_loader(opt.data_path, 'test_data', vocab, opt,
                                     batch_size, False, workers)
    return test_loader
