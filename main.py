import albumentations
import cv2
import joblib
import numpy as np
import torch

from src.models import CustomModel

# Loading saved Label Encoder Model
lb = joblib.load("lb.pkl")

LABELS = lb.classes_
NUM_CLASSES = len(LABELS)
MODEL = "saved_models/model-train-acc-98val-acc-99.pth"
IMAGE_SIZE = 128


def main(model):
    cap = cv2.VideoCapture(0)
    if (cap.isOpened() == False):
        print('Error while trying to open camera. Plese check again...')

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(
        'outputs/asl.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        cv2.rectangle(frame, (100, 100), (324, 324), (20, 34, 255), 2)
        hand = frame[100:324, 100:324]

        image = np.array(hand)

        augmentations = albumentations.Compose([
            albumentations.Resize(128, 128, always_apply=True),
            albumentations.Normalize(always_apply=True)
        ], p=1)
        augmented = augmentations(image=image)
        image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float).cuda()
        image = image.unsqueeze(0)
        outputs = model(image)
        preds = torch.argmax(outputs.data, 1)

        cv2.putText(frame, LABELS[preds], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow('image', frame)
        out.write(frame)

        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = CustomModel(num_classes=28).cuda()
    model.load_state_dict(torch.load(MODEL))
    main(model)
