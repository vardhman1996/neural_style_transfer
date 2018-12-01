import cv2
from settings import *
import torch
import os
import argparse
from torchvision import transforms
from utils import makedir


def load_model(filename):
    if USE_CUDA:
        model = torch.load(filename)
    else:
        model = torch.load(filename, map_location='cpu')
    return model.to(DEVICE)

def get_loader():
    loader = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(IMSIZE),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255.0))
            ])
    return loader

def unload_image(frame):
    img = frame.cpu().clone().clamp(0, 255).numpy()
    img = img.squeeze(0)  # remove the fake batch dimension
    img = img.transpose(1, 2, 0).astype("uint8")

    permute = [2, 1, 0]
    img = img[:, :, permute]
    return img


def process_frame(frame, model, loader):
    frame = cv2.resize(frame, (IMSIZE, IMSIZE))

    frame = loader(frame)
    frame = frame.unsqueeze(0)

    with torch.no_grad():
        frame = model(frame)

    frame = unload_image(frame)

    return frame

def eval_model_live(model_path):
    model = load_model(model_path)
    loader = get_loader()

    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        frame = process_frame(frame, model, loader)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def eval_model_offline(model_path, video_file, output_file_dir, model_number):
    model = load_model(model_path)
    loader = get_loader()

    video_path = os.path.join(VIDEO_INPUT, video_file)
    video_out = os.path.join(output_file_dir, "output-{}.avi".format(model_number))

    cap = cv2.VideoCapture(video_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_out, fourcc, input_fps, (IMSIZE, IMSIZE))

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames: ", num_frames)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            i += 1
            frame = process_frame(frame, model, loader)
            out.write(frame)

            if (i + 1) % 10 == 0:
                print("{} out of {} completed".format(i + 1, num_frames))
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


main_arg_parser = argparse.ArgumentParser()
main_arg_parser.add_argument("--name", type=str, required=True)
main_arg_parser.add_argument("--style", type=str, required=True)
main_arg_parser.add_argument("--model_no", type=str, required=True)
main_arg_parser.add_argument("--video_file", type=str, required=False)
args = main_arg_parser.parse_args()

style_path = os.path.join(args.style, args.name)
checkpoint_path = os.path.join(CHECKPOINT_PATH, style_path)
model_path = os.path.join(checkpoint_path, "{}.model".format(args.model_no))

if args.video_file:
    output_file_dir = os.path.join(VIDEO_OUT, args.style)
    makedir(output_file_dir)
    eval_model_offline(model_path, args.video_file, output_file_dir, args.model_no)
else:
    eval_model_live(model_path)
