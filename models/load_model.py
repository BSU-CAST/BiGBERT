import gdown
import pickle
import tensorflow as tf
from pathlib import Path
from keras.models import load_model
from keras_self_attention import SeqSelfAttention  # pip install keras-self-attention==0.42.0


def load_bigru():
    new_model = load_model("../models/bigru.h5", custom_objects={"SeqSelfAttention": SeqSelfAttention})
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.2)
    new_model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
    return new_model


def load_bert_with_edu():
    bertedu_path = "../models/bertedu_1e-6lr.p"
    if not Path(bertedu_path).exists():
        bertedu_pub_storage = "https://drive.google.com/uc?id=116pGILUWd9m4QFCbWJnlP8UdVBtCGVny"
        gdown.download(bertedu_pub_storage, bertedu_path, quiet=False)


    return pickle.load(open("../models/bertedu_1e-6lr.p", "rb"))


def load_bigbert():
    built_model = load_model("../models/bigbert.h5")
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.2)
    built_model.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    return built_model
