import torch

from captcha import *

paths = []
labels = []
for image in os.listdir(DIR):
    paths.append(os.path.join(DIR, image))
    labels.append(image.split(".")[0])

df = pd.DataFrame({
    "paths": paths,
    "labels": labels
})

df.head()

def get_image(image_path=None):
    if image_path == None:
        img = random.choice(df["paths"])
        return [img]
    return [image_path]


def get_sample_photo(image_path=None):
    img = get_image(image_path)
    eval_dataset = DatasetClassifier(
        image_paths=img, targets=[np.array([x for x in np.arange(10)])], resize=(IMG_HEIGHT, IMG_WIDTH)
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False
    )

    return img, eval_loader


def predict_function(model, data_loader):
    model.eval()
    fin_preds = []
    with torch.no_grad():
        for data in data_loader:
            for k, v in data.items():
                data[k] = v.to(DEVICE)

            batch_preds, _ = model(**data)
            fin_preds.append(batch_preds)

        return fin_preds


image_path, eval_loader = get_sample_photo()
print(image_path)


def clean_decoded_predictions(unclean_predictions):
    cleaned_predictions = []
    for i in unclean_predictions:
        if i != "-":
            cleaned_predictions.append(i)

    cleaned_predictions = "".join(cleaned_predictions)

    if len(cleaned_predictions) == 10:
        return cleaned_predictions

    else:
        prev = "-"
        new_cleaned_predictions = []
        for char in cleaned_predictions:
            if char == prev:
                continue
            new_cleaned_predictions.append(char)
            prev = char
        res = "".join(new_cleaned_predictions)
        return res


def predict_captcha(model, eval_loader, image_path):
    plt.figure(figsize=(15, 5))
    image = mpimg.imread(image_path[0])
    target = image_path[0].split("/")[-1].split(".")[0]
    plt.title(image_path[0].split("/")[-1])
    plt.imshow(image)

    valid_preds = predict_function(model, eval_loader)
    for vp in valid_preds:
        current_preds = decode_predictions(vp, lbl_enc)

    preds = clean_decoded_predictions(current_preds[0])

    success = True if preds == target else False

    return {
        "success": success,
        "prediction": preds,
        "real": target
    }

model = MyCaptchaModel(num_chars=len(lbl_enc.classes_))



preds = predict_captcha(model, eval_loader, image_path)
print(preds)