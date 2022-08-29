import tensorflow as tf
import pandas as pd
from transformers import PushToHubCallback, \
    TFAutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer
from datasets import load_dataset, Dataset

BATCHSIZE = 128

model_checkpoint = "microsoft/xtremedistil-l6-h256-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)




#data_files = {"train": "no_split_only_1_min07/topic_3_labelsonly70.csv"}

dataset = load_dataset("yahoo_answers_topics")
#topiclabel = ["work","news","sports","music","movies","politics","phones","self-driving cars","family","cars","climate change","languages","business","health","science","style","opinion","economy","history","technology","affair","development","mobility"]

#dataset = dataset.map(lambda x : {"labels": [x[c] for c in topiclabel]})
print(dataset)

#dataset = dataset.remove_columns(['question_title'])
dataset.set_format(type="pandas")
test = dataset["test"][:]
test['fullque'] = test['question_title'] + ' ' + test['question_content']
print(test.head())
testans = test[["topic","best_answer"]]
testans = testans.rename(columns ={"best_answer": "text"})
testqu = test[["topic","fullque"]]
testqu = testqu.rename(columns ={"fullque": "text"})
train = dataset["train"][:]
train["fullque"] = train["question_title"]+ ' ' + train['question_content']
trainans = train[["topic","best_answer"]]
trainans = trainans.rename(columns ={"best_answer": "text"})
trainqu = train[["topic","fullque"]]
trainqu = trainqu.rename(columns ={"fullque": "text"})

dataset = Dataset.from_pandas(pd.concat([testans, testqu,trainans,trainqu]))


#print(dataset)


#dataset = dataset["train"]
#datasetval = Dataset.from_pandas(pd.concat([emval, dataset["validation"][:]]))
#datasettest = Dataset.from_pandas(pd.concat([emtest, dataset["test"][:]]))

print(dataset[0])
exit(0)
tokenized_train_datasets = dataset.map(tokenize_function, batched=True, batch_size=None)

#tokenized_test_datasets = dataset["test"].map(tokenize_function, batched=True, batch_size=None)
#tokenized_val_datasets = datasetval.map(tokenize_function, batched=True, batch_size=None)
#tokenized_train_datasets = tokenized_train_datasets.map(maplable)
#tokenized_test_datasets = tokenized_test_datasets.map(maplable)
#tokenized_val_datasets = tokenized_val_datasets.map(maplable)
print(tokenized_train_datasets[0])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
# small_eval_dataset = tokenized_test_datasets.shuffle(seed=42).select(range(1000))
tf_train_dataset = tokenized_train_datasets.to_tf_dataset(
    columns=tokenizer.model_input_names,
    label_cols=["topic"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=BATCHSIZE,
)
# tf_test_dataset = tokenized_test_datasets.to_tf_dataset(
#     columns=tokenizer.model_input_names,
#     label_cols=["topic"],
#     shuffle=True,
#     collate_fn=data_collator,
#     batch_size=BATCHSIZE,
# )


model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                             num_labels=10, problem_type="multi_label_classification")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=tf.metrics.SparseCategoricalAccuracy()
              )

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=3,  # Stop after 3 epochs of no improvement
    monitor='val_loss',  # Look at validation_loss
    min_delta=0.01,  # After 0 change
    mode='min',  # Stop when quantity has stopped decreasing
    verbose=1

)

push_to_hub_callback = PushToHubCallback(

    output_dir="./huggingface_topic04", tokenizer=tokenizer,
    hub_model_id="jonaskoenig/topic_classification_04",
    hub_token="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",

)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs', )
model.fit(tf_train_dataset, epochs=10, batch_size=BATCHSIZE,
          callbacks=[push_to_hub_callback, tensorboard, early_stopping])
#_, accuracy = model.evaluate(tf_test_dataset)
#print(f"Test accuracy: {round(accuracy * 100, 2)}%")
model.push_to_hub("jonaskoenig/topic_classification_04",
                  use_auth_token="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")
tokenizer.push_to_hub('jonaskoenig/topic_classification_04',
                      use_auth_token="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")
# tensorboard.push_to_hub('jonaskoenig/xtremedistil-l6-h384-uncased-future-time-references',
#                      use_auth_token="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")
