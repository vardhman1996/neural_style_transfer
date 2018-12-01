from settings import *
import torch.optim as optim
import torch.nn.functional as F
import image_transform_net as img_net
import vgg_net
import data_preprocessing as dp
import utils
import multiprocessing
import os
import argparse

style_img = dp.image_loader("./images/styles/{}.jpg".format(STYLE)).unsqueeze(0)
LOGGER.log_image(file_path="./images/styles/{}.jpg".format(STYLE), file_name=STYLE)
LOGGER.log_image(file_path="./images/content/{}.jpg".format(EVAL_CONTENT_IMAGE), file_name=EVAL_CONTENT_IMAGE)


def train(name, content_weight=1e2, style_weight=1e10):
    data_train = dp.ImageFolderLoader(IMAGE_FOLDER)

    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if USE_CUDA else {}
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=HYPERPARAMETERS['batch_size'], shuffle=True, **kwargs)

    vgg_loss_net = vgg_net.VGG_loss_net()
    image_transform_net = img_net.ImageTransformation().to(DEVICE)
    optimizer = optim.Adam(image_transform_net.parameters(), lr=HYPERPARAMETERS['learning_rate'])

    style_batch = style_img.repeat(HYPERPARAMETERS['batch_size'], 1, 1, 1).to(DEVICE)
    style_features = vgg_loss_net(style_batch)
    gram_style = [utils.gram_matrix(y) for y in style_features]

    style_path = os.path.join(STYLE, name)
    checkpoint_path = utils.make_checkpoint_dir(style_path)
    prediction_path = utils.make_checkpoint_dir(os.path.join(style_path, "predictions"))

    total_loss_log = []
    style_loss_log = []
    content_loss_log = []
    index_log = []
    log = {'total_loss_log':total_loss_log, 'style_loss_log':style_loss_log, 'content_loss_log':content_loss_log, 'index_log':index_log}
    log_filename = os.path.join(checkpoint_path, "log.pkl")

    for e in range(HYPERPARAMETERS['epochs']):
        image_transform_net.train()
        print("Epochs {}".format(e))
        for batch_idx, content_batch in enumerate(train_loader):
            optimizer.zero_grad()
            content_batch = content_batch.to(DEVICE)
            y_hat_batch = image_transform_net(content_batch)

            total_loss, style_loss, content_loss = loss_evaluation(vgg_loss_net, content_batch, y_hat_batch, gram_style,
                                                                   content_weight=content_weight, style_weight=style_weight)
            total_loss.backward(retain_graph=True)
            if (batch_idx + 1) % ITERS == 0:
                print("Itr {} Total loss {} ".format(batch_idx + 1, total_loss.item() / HYPERPARAMETERS['batch_size']))
                # print("\t Style loss: {} Content loss: {}".format(style_loss.item(), content_loss.item()))
                eval(image_transform_net, filename=os.path.join(prediction_path, "test_{}_{}.jpg".format(e, batch_idx + 1)))

                # COMET ML
                step = e * len(train_loader) + batch_idx
                LOGGER.log_metric("total_loss", total_loss.item(), step=step)
                LOGGER.log_metric("style_loss", style_loss.item(), step=step)
                LOGGER.log_metric("content_loss", content_loss.item(), step=step)

                log['total_loss_log'].append(total_loss.item())
                log['style_loss_log'].append(style_loss.item())
                log['content_loss_log'].append(content_loss.item())
                log['index_log'].append(e * len(train_loader) + batch_idx)
                utils.write_log(log, log_filename)
            optimizer.step()
        torch.save(image_transform_net, os.path.join(checkpoint_path, "{}.model".format(e)))
        eval(image_transform_net, filename=os.path.join(prediction_path, "test_epoch_model_{}.jpg".format(e)))
        LOGGER.log_epoch_end(e)

    utils.plot_log(log_filename)
    return image_transform_net


def eval(model, content_img=None, filename='test'):
    with torch.no_grad():
        if content_img is None:
            content_img = dp.image_loader("./images/content/{}.jpg".format(EVAL_CONTENT_IMAGE)).unsqueeze(0).to(DEVICE)

        output = model(content_img)
        dp.imshow(output, filename=filename)
        LOGGER.log_image(file_path=filename, file_name=os.path.basename(filename))


def loss_evaluation(vgg_loss_net, content_batch, y_hat_batch, gram_style, content_weight=1e2, style_weight=1e10):
    batch_size = len(content_batch)

    content_features = vgg_loss_net(content_batch)
    y_hat_features = vgg_loss_net(y_hat_batch)

    content_loss = content_weight * F.mse_loss(y_hat_features.relu2_2, content_features.relu2_2)

    style_loss = 0.0
    for y_hat_ft, gm_style in zip(y_hat_features, gram_style):
        gm_y_hat = utils.gram_matrix(y_hat_ft)
        style_loss += F.mse_loss(gm_y_hat, gm_style[:batch_size, :, :])

    style_loss *= style_weight
    total_loss = content_loss + style_loss
    return total_loss, style_loss, content_loss


main_arg_parser = argparse.ArgumentParser()
main_arg_parser.add_argument("--name", type=str, required=True)
main_arg_parser.add_argument("--content_weight", type=str, required=True)
main_arg_parser.add_argument("--style_weight", type=str, required=True)
args = main_arg_parser.parse_args()
content_weight = float(args.content_weight)
style_weight = float(args.style_weight)

HYPERPARAMETERS["content_weight"] = content_weight
HYPERPARAMETERS["style_weight"] = style_weight
LOGGER.log_multiple_params(HYPERPARAMETERS)

model = train(args.name, content_weight=content_weight, style_weight=style_weight)
