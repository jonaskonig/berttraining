import tensorflow as tf

from transformers import DistilBertTokenizer, TFDistilBertModel, DefaultDataCollator, PushToHubCallback, \
    TFAutoModelForSequenceClassification, TFDistilBertForSequenceClassification, DataCollatorWithPadding, AutoTokenizer
from datasets import load_dataset, DatasetDict

BATCHSIZE = 128
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

model_checkpoint = "microsoft/xtremedistil-l6-h256-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_function(examples):
    return tokenizer(examples["doc"], padding="max_length", truncation=True,max_length=64)


push_to_hub_callback = PushToHubCallback(

    output_dir="./huggingfacedest8", tokenizer=tokenizer, hub_model_id="jonaskoenig/xtremedistil-l6-h256-uncased-question-vs-statement-classifier",


)

access_token = "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"
# dataset = load_dataset("jonaskoenig/twitter_statements_tenses", data_files=data_files, use_auth_token=access_token, )
#data_files = {"train":["output/statements_sh_bal_train.csv","output/trumptrain.csv"], "test":["output/statements_sh_bal_test.csv","output/trumptest.csv" ],"valid":["output/statements_sh_bal_val.csv","output/trumpval.csv"]}
data_files = {"train": ["train.csv","val.csv"], "test": "test.csv"}
dataset = load_dataset('jonaskoenig/Questions-vs-Statements-Classification',use_auth_token=access_token, data_files=data_files)
dataset = dataset.remove_columns("Unnamed: 0")
print(dataset["train"][0])
tokenized_train_datasets = dataset["train"].map(tokenize_function, batched=True, batch_size=None)
tokenized_test_datasets = dataset["test"].map(tokenize_function, batched=True, batch_size=None)
#tokenized_val_datasets = dataset["valid"].map(tokenize_function, batched=True, batch_size=None)

print(tokenized_train_datasets)
# small_train_dataset = tokenized_train_datasets.shuffle(seed=42).select(range(1000))
# data_collator = DefaultDataCollator(return_tensors="tf")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
# small_eval_dataset = tokenized_test_datasets.shuffle(seed=42).select(range(1000))
tf_train_dataset = tokenized_train_datasets.to_tf_dataset(
    columns=tokenizer.model_input_names,
    label_cols=["target"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=BATCHSIZE,
)

tf_validation_dataset = tokenized_test_datasets.to_tf_dataset(
    columns=tokenizer.model_input_names,
    label_cols=["target"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=BATCHSIZE,
)
#tf_test_dataset = tokenized_val_datasets.to_tf_dataset(columns=tokenizer.model_input_names,
#                                                       label_cols=['ft_tense'], shuffle=False, batch_size=BATCHSIZE,
#                                                       collate_fn=data_collator
#                                                       )

# print(tf_train_dataset)

model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                              num_labels=2,problem_type="multi_label_classification")
# model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)
# model.compile(

#    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),

#    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

#    metrics=tf.metrics.SparseCategoricalAccuracy(),

# )
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=tf.metrics.SparseCategoricalAccuracy()
              )

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="finetuned-xtremedistil-265",
    save_best_only=True
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=3,  # Stop after 3 epochs of no improvement
    monitor='val_loss',  # Look at validation_loss
    min_delta=0.01,  # After 0 change
    mode='min',  # Stop when quantity has stopped decreasing
    verbose=1

)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs', )
model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=5, batch_size=BATCHSIZE,
          callbacks=[push_to_hub_callback, tensorboard,early_stopping])
#_, accuracy = model.evaluate(tf_test_dataset)
#print(f"Test accuracy: {round(accuracy * 100, 2)}%")

model.push_to_hub("jonaskoenig/xtremedistil-l6-h256-uncased-question-vs-statement-classifier",
                  use_auth_token="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")
tokenizer.push_to_hub('jonaskoenig/xtremedistil-l6-h256-uncased-question-vs-statement-classifier',
                      use_auth_token="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")
