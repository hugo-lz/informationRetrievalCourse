#-------------------------------------------------------------------------------
# Ejemplo de traductor de texto usando una red LSTM. Es una demostración de como el uso de embeddings y contexto sirve para traducir texto
# Los embeddings aportan representación semántica de las palabras, permitiendo a la red identificar mejor su significado. 
# La red LSTM añade procesamiento secuencial de las frases, lo que le permite aprender características que dependen del orden.
#La red es demasiado pequeña y se entrena demasiado poco como para resolver apropiadamente el problema,
#solo es un ejemplo simple de como funciona la arquitectura de estos sistemas.
#-------------------------------------------------------------------------------

# Importaciones requeridas.
import commonFunctions, os, numpy as np
from commonFunctions import Chronometer, set_random_seed, Sequential, Dense, Embedding, LSTM, RepeatVector, TimeDistributed, RMSprop, saveResults
from textTranslator__DataReader import dataReader

#-------------------------------------------------------------------------------
# Modelo LSTM pequeño con embeddings que modela un encoder/decoder. La capa de embeddings necesita conocer el tamaño del 
# vocabulario de la colección, las dimensiones de los embeddings que tiene que generar y la longitud máxima de las frases a codificar.
# La capa repeatVecror pasa los estados ocultos en cada etapa del decoder. La capa final da un valor (palabra) para cada ejecución temporal del decoder
# Al querer generar un vector de números con los índices de las palabras en el lenguaje destino en vez de un one-hot la función de pérdida es sparse_categorical_crossentropy
# Es un modelo demasiado pequeño para funcionar correctamente, pero da una idea de la arquitectura usada.
#-------------------------------------------------------------------------------
def define_model(X_tamVoc,y_tamVoc,y_maxlen):
    EMBEDDINGS_SIZE = 50 # Número de dimensiones del vector de Embeddings
    model = Sequential()
    model.add(Embedding(X_tamVoc, EMBEDDINGS_SIZE, mask_zero=True)) # mask_zero hace que se cree una máscara indicando que posiciones son padding, esto evita que aprenda patrones de las casillas vacías.
    model.add(LSTM(64))
    # El encoder ha generado una representación de la frase. Esta representación hay que proporcionarsela al decoder tantas veces como tamaño de la frase de salida.
    # El decoder coge esta representación (entrada) + su estado y genera la siguiente palabra y el siguiente estado.
    model.add(RepeatVector(y_maxlen)) 
    model.add(LSTM(64, return_sequences=True)) # return_sequences hace que en cada iteración de la LSTM de una salida. Esto es necesario ya que queremos generar tantas palabras como tamaño de la salida.
    model.add(TimeDistributed(Dense(y_tamVoc, activation='softmax'))) # Para la salida de cada palabra aplicamos una red densa que nos identifica una palabra del vocabulario de salida
    model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
    return model

#-------------------------------------------------------------------------------
# Carga de datos, entrenamiento y test con el modelo anteriormente definido.
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    dir='results/textTranslator_LSTM'
    set_random_seed(0) # Fijamos las semillas de los generadores de números aleatorios usados para tener reproducibilidad.
    os.makedirs(dir, exist_ok=True) # Creamos una carpeta donde guardar los resultados del entrenamiento y test
    
    # Los ficheros de entrenamiento y test a leer son fijos en la práctica. La capa de embeddings requiere enteros no normalizados.
    X_train, y_train, X_test, y_test, X_tokenizer, y_tokenizer = dataReader(0.1, 10)
    model = define_model(len(X_tokenizer.word_index), len(y_tokenizer.word_index), len(y_train[0]))
    
    print('----------------------------------------------------')
    print('Training the model')
    print('----------------------------------------------------')
    print('-     -     -     -     -     -     -     -     -     ')
    # Entrenamos el modelo con los datos disponibles.
    with Chronometer() as chronometer:
        history = model.fit(X_train, y_train.reshape(y_train.shape[0], y_train.shape[1], 1),
                        epochs=5, batch_size=512, validation_split=0.2, verbose=1)

    # Guardamos el modelo y los resultados del entrenamiento.
    saveResults(model, history, chronometer.message, X_test, y_test.reshape(y_test.shape[0], y_test.shape[1], 1), dir)
    with open(dir+'/testResults.txt', 'a', encoding='utf-8', errors='ignore') as f:
        printAll = lambda *args: (print(*args), print(*args, file=f))
        # Ejemplo de una traducción
        printAll("Translation example")
        prediction = np.argmax(model.predict(np.expand_dims(X_test[0], axis=0), verbose=0)[0], axis=1)
        printAll("English text: ",X_tokenizer.sequences_to_texts([X_test[1000]]))
        printAll("Real Spanish translation: ", y_tokenizer.sequences_to_texts([y_test[1000]]))
        printAll("Predicted Spanish translation: ",y_tokenizer.sequences_to_texts([prediction]))
        printAll("")
        

