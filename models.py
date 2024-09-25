from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, SimpleRNN



def compile_model(name, train_input):

    # Define the LSTM model
    model = Sequential(name=name)
    model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(train_input.shape[1], train_input.shape[2])))

    model.add(Dropout(0.2))  # Adding dropout for regularization
    model.add(Bidirectional(LSTM(128, return_sequences=True)))

    # model.add(Dropout(0.2))  # Adding dropout for regularization
    # model.add((LSTM(128, return_sequences=True)))

    model.add(Dropout(0.2))  # Adding dropout for regularization
    model.add(Bidirectional(LSTM(128, return_sequences=True)))

    # model.add(Dropout(0.2))  # Adding dropout for regularization
    # model.add((LSTM(128, return_sequences=True)))

    model.add(Dropout(0.2))  # Adding dropout for regularization
    model.add(Bidirectional(LSTM(64)))

    model.add(Dense(1, activation='sigmoid'))
    # model.add(Dense(1, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Display the model summary
    # model.summary()

    return model




def compile_LSTM(name, train_input):

    # Define the LSTM model
    model = Sequential(name=name)
    model.add(LSTM(256, return_sequences=True, input_shape=(train_input.shape[1], train_input.shape[2])))

    model.add(Dropout(0.2))  # Adding dropout for regularization
    model.add(LSTM(128, return_sequences=True))

    model.add(Dropout(0.2))  # Adding dropout for regularization
    model.add(LSTM(128, return_sequences=True))

    model.add(Dropout(0.2))  # Adding dropout for regularization
    model.add(LSTM(64))

    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Display the model summary
    # model.summary()

    return model


def compile_RNN(name, train_input):

    # Define the RNN model
    model = Sequential(name=name)
    model.add(SimpleRNN(256, return_sequences=True, input_shape=(train_input.shape[1], train_input.shape[2])))

    model.add(Dropout(0.2))  # Adding dropout for regularization
    model.add(SimpleRNN(128, return_sequences=True))

    model.add(Dropout(0.2))  # Adding dropout for regularization
    model.add(SimpleRNN(128, return_sequences=True))

    model.add(Dropout(0.2))  # Adding dropout for regularization
    model.add(SimpleRNN(64))

    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Display the model summary
    # model.summary()

    return model

