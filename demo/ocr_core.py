import torch
import os
from train import id_to_string

from checkpoint import load_checkpoint
from torchvision import transforms
from dataset import LoadEvalDataset, collate_eval_batch

from utils import get_network
import csv
from torch.utils.data import DataLoader
from flags import Flags
import random
from tqdm import tqdm


class ImageTransform():
    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.Resize((100, 450)),
            transforms.ToTensor(),
        ])

    def __call__(self, img):
        return self.data_transform(img)

@torch.no_grad()
def ocr_core(image):
    """
    This function will handle the core OCR processing of images.
    """
    is_cuda = torch.cuda.is_available()
    checkpoint = load_checkpoint("./log/satrn/checkpoints/0050.pth", cuda=is_cuda)
    options = Flags(checkpoint["configs"]).get()

    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)

    model_checkpoint = checkpoint["model"]

    trans = ImageTransform()
    image = image.convert("L")
    image = trans(image)

    #Dummy Dataset & loader for model call
    transformed = transforms.Compose(
        [
            transforms.Resize((options.input_size.height, options.input_size.width)),
            transforms.ToTensor(),
        ]
    )
    dummy_gt = "\sin " * 230  # set maximum inference sequence = 230
    eval_dir = os.environ.get('SM_CHANNEL_EVAL', './dummy_data')
    file_path = os.path.join(eval_dir, 'input.txt')
    root = os.path.join(os.path.dirname(file_path), "images")
    with open(file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data] # x[0] == train_00000.jpg
    test_dataset = LoadEvalDataset(
        test_data, checkpoint["token_to_id"], checkpoint["id_to_token"], crop=False, transform=transformed,
        rgb=1
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        collate_fn=collate_eval_batch,
    )

    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        test_dataset,
    )
    model.eval()

    for d in test_data_loader:
        image = image.to(device) #target image
        image = torch.unsqueeze(image, 0)
        dummy_expected = d["truth"]["encoded"].to(device) #dummy expected
        output = model(image, dummy_expected, False, 0.0)
        #print("output",output.shape)
        decoded_values = output.transpose(1, 2)
        #print("decoded_values", decoded_values.shape)
        _, sequence = torch.topk(decoded_values, 1, dim=1)
        #print("extracted_sequence", sequence.shape)
        sequence = sequence.squeeze(0)
        #print("sqeueezed_sequence", sequence.shape)
        sequence_str = id_to_string(sequence, test_data_loader, do_eval=1)
        #print(sequence_str)
        break #single img inference
    text = sequence_str[0]
    return text
