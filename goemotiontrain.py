import tensorflow as tf
import pandas as pd
from transformers import PushToHubCallback, \
    TFAutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer
from datasets import load_dataset, Dataset

BATCHSIZE = 128

model_checkpoint = "microsoft/xtremedistil-l6-h256-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def maplable(example):
    arr = [0] * 28
    for k in example["labels"]:
        arr[k] = 1
    example["labels"] = arr
    return example


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)


translate = {0: 25, 1: 17, 2: 18, 3: 2, 4: 14, 5: 26}


def assignnewlabel(lab):
    newlab = translate[lab]
    arr = [0] * 28
    arr[newlab]= 1
    return arr


dataset = load_dataset("go_emotions", "raw")
emotionlabel = [
 'admiration',
 'amusement',
 'anger',
 'annoyance',
 'approval',
 'caring',
 'confusion',
 'curiosity',
 'desire',
 'disappointment',
 'disapproval',
 'disgust',
 'embarrassment',
 'excitement',
 'fear',
 'gratitude',
 'grief',
 'joy',
 'love',
 'nervousness',
 'optimism',
 'pride',
 'realization',
 'relief',
 'remorse',
 'sadness',
 'surprise',
 'neutral']
dataset = dataset.map(lambda x : {"labels": [x[c] for c in emotionlabel]})
cols = dataset["train"].column_names
cols.remove("labels")
cols.remove("text")
dataset = dataset.remove_columns(cols)

emotions = load_dataset("emotion")
emotions.set_format(type="pandas")
dataset.set_format(type="pandas")
emtrain = emotions["train"][:]
print(emtrain.head())
emtrain["labels"] = emtrain["label"].apply(assignnewlabel)
emtrain.pop("label")
emtest = emotions["test"][:]
emtest["labels"] = emtest["label"].apply(assignnewlabel)
emtest.pop("label")
emval = emotions["validation"][:]
emval["labels"] = emval["label"].apply(assignnewlabel)
emval.pop("label")
print(emval.head())
datasettrain = Dataset.from_pandas(pd.concat([emtrain, dataset["train"][:],emval,emtest]))
#datasetval = Dataset.from_pandas(pd.concat([emval, dataset["validation"][:]]))
#datasettest = Dataset.from_pandas(pd.concat([emtest, dataset["test"][:]]))

print(datasettrain)
tokenized_train_datasets = datasettrain.map(tokenize_function, batched=True, batch_size=None)
#tokenized_test_datasets = datasettest.map(tokenize_function, batched=True, batch_size=None)
#tokenized_val_datasets = datasetval.map(tokenize_function, batched=True, batch_size=None)
#tokenized_train_datasets = tokenized_train_datasets.map(maplable)
#tokenized_test_datasets = tokenized_test_datasets.map(maplable)
#tokenized_val_datasets = tokenized_val_datasets.map(maplable)
print(tokenized_train_datasets[0])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
# small_eval_dataset = tokenized_test_datasets.shuffle(seed=42).select(range(1000))
tf_train_dataset = tokenized_train_datasets.to_tf_dataset(
    columns=tokenizer.model_input_names,
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=BATCHSIZE,
)



model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                             num_labels=28, problem_type="multi_label_classification")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
              loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=True),
              metrics=tf.metrics.BinaryCrossentropy()
              )

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=3,  # Stop after 3 epochs of no improvement
    monitor='val_loss',  # Look at validation_loss
    min_delta=0.01,  # After 0 change
    mode='min',  # Stop when quantity has stopped decreasing
    verbose=1

)

push_to_hub_callback = PushToHubCallback(

    output_dir="./huggingfacegoemotion", tokenizer=tokenizer,
    hub_model_id="jonaskoenig/xtremedistil-l6-h256-uncased-go-emotion",
    hub_token="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",

)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs', )
model.fit(tf_train_dataset, epochs=10, batch_size=BATCHSIZE,
          callbacks=[push_to_hub_callback, tensorboard, early_stopping])
#_, accuracy = model.evaluate(tf_test_dataset)
#print(f"Test accuracy: {round(accuracy * 100, 2)}%")
model.push_to_hub("jonaskoenig/xtremedistil-l6-h384-uncased-future-time-references",
                  use_auth_token="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")
tokenizer.push_to_hub('jonaskoenig/xtremedistil-l6-h384-uncased-future-time-references',
                      use_auth_token="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")
# tensorboard.push_to_hub('jonaskoenig/xtremedistil-l6-h384-uncased-future-time-references',
#                      use_auth_token="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")
