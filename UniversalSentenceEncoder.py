import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class UniversalSentenceEncoder(object):
    def __init__(self,
                 module_url="https://tfhub.dev/google/universal-sentence-encoder-large/2",
                 max_batch_size=10000
                 ):
        # To prevent memory overflow error on very large number of messages/batch
        self.max_batch_size = max_batch_size

        # Hub module load
        self.embed = hub.Module(module_url)
        self.input_placeholder = tf.placeholder(tf.string, shape=(None))
        self.message_encodings = tf.nn.l2_normalize(self.embed(self.input_placeholder))

        # Creating session forever
        self.session = tf.Session()
        with self.session.as_default():
            # and initialising models variables for once and all
            self.session.run(tf.global_variables_initializer())
            self.session.run(tf.tables_initializer())

    def get_sentence_vector(self, messages):

        print("Calculating vectors")
        messages_embedding = []
        len_messages = len(messages)

        with self.session.as_default():

            counter = 0
            while True:
                end = min(len_messages, counter + self.max_batch_size)

                messages_embedding.append(self.session.run(self.message_encodings,
                                                           feed_dict={
                                                               self.input_placeholder: messages[counter: end]
                                                           }))
                counter = end
                if counter == len_messages:
                    break
        print("Done calculating vectors")
        return np.array(messages_embedding)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
        super(UniversalSentenceEncoder, self).__exit__()


if __name__ == '__main__':
    obj = UniversalSentenceEncoder()
    msgs = ["Sup?", "Hi!!"]
    print((obj.get_sentence_vector(msgs)))
