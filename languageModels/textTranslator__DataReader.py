#-------------------------------------------------------------------------------
# Método de lectura y procesamiento de datos usados en el traductor de texto.
# Incluye las funciones auxiliares usadas por este método.
#-------------------------------------------------------------------------------

# Importaciones requeridas.
import pandas as pd
from commonFunctions import tokenizeText, cleanTexts

#-------------------------------------------------------------------------------
# Método para leer los ficheros tabulares del ejercicio de traducción de texto (Ingles, Español y Licencia)
#-------------------------------------------------------------------------------
def __readDataframe(file):
    df = pd.read_csv(file, index_col=False, delimiter='\t', names=['Ingles', 'Espanol', 'Licencia'])
    df.drop(['Licencia'], axis=1, inplace=True)
    return df

#-------------------------------------------------------------------------------
# Método que devuelve los datos de entrenamiento y test del traductor de texto.
# Lee los datos, los limpia, y segmenta. Las categorías las convierte a representación one-hot.
# Selecciona un fracción de los datos (aleatorio). Esto esta hecho en el ejercicio para que el entrenamiento sea más rápido a costa de precisión.
#-------------------------------------------------------------------------------
def dataReader(fraction=1,  maxLength=200):
    # Cargamos los datos de entrenamiento y test.
    dataset = __readDataframe('datos/traductorFrasesEnEs.txt')
    dataset = dataset.sample(frac=fraction, random_state=0)
    
    #Limpiamos los textos de caracteres no alfanuméricos.
    X_data = cleanTexts(dataset['Ingles'].values)
    y_data = cleanTexts(dataset['Espanol'].values)
    
    #Dividimos los datos en entrenamiento y test y convertimos el texto en secuencias numéricas. 
    # En este caso el objetivo (y) también son cadenas de texto.
    divIndex = int(len(X_data) * 0.8)
    X_train, X_test, tx = tokenizeText(X_data[:divIndex], X_data[divIndex:], maxLength)
    y_train, y_test, ty = tokenizeText(y_data[:divIndex], y_data[divIndex:], maxLength)
    
    # Se devuelven las colecciones de entrenamiento y test, y los tokenizadores del el lenguaje original y traducido.                
    return X_train, y_train, X_test, y_test, tx, ty