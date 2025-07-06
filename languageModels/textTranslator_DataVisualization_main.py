#-------------------------------------------------------------------------------
# Código de ejemplo para entender el contenido de la colección de datos para entrenar el traductor de texto
# y las estructuras generadas por el código relacionado con la carga y procesamiento de los datos.
#-------------------------------------------------------------------------------

# Importaciones requeridas.
import pandas as pd, commonFunctions
from commonFunctions import Tokenizer, pad_sequences, cleanTexts, tokenizeText

#-------------------------------------------------------------------------------
# Realiza la carga de los datos en pandas, su limpieza y segmentación, mostrando ejemplos por pantalla.
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    #La estructura de los datos es un fichero tabular con texto en ingles, traducción al español y licencia
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('datos/traductorFrasesEnEs.txt', index_col=False, delimiter='\t', names=['Ingles', 'Espanol', 'Licencia'])
    print('---------------------------------------------------')
    print('First and last data rows')
    print('---------------------------------------------------')
    pd.set_option('display.width', 200)
    print(df.head(5));  print(df.tail(5))
    
    # El proceso de limpieza deja solo caracteres útiles para el proceso de aprendizaje.
    X_cleanedTexts = cleanTexts(df['Ingles'].values)
    y_cleanedTexts = cleanTexts(df['Espanol'].values)
    print('\n---------------------------------------------------')
    print('Example of data cleaning (before/after)')
    print('---------------------------------------------------')
    print(df['Ingles'][10000], df['Espanol'][10000])
    print(X_cleanedTexts[10000], y_cleanedTexts[10000])
    
    # El proceso de segmentación crea un vector de palabras ajustado para que tengan todos el mismo tamaño con codificación numérica.
    X_tokenizedText, _, X_tokenizer = tokenizeText(X_cleanedTexts, [[]], 10)
    y_tokenizedText, _, y_tokenizer = tokenizeText(y_cleanedTexts, [[]], 10)

    print('\n---------------------------------------------------')
    print('Result of tokenization of the previous texts')
    print('---------------------------------------------------')
    print('English text:',X_cleanedTexts[10000])
    print(X_tokenizedText[10000])
    print(list(X_tokenizer.word_index.items())[:10])
    print('Spanish translation:',y_cleanedTexts[10000])
    print(y_tokenizedText[10000])
    print(list(y_tokenizer.word_index.items())[:10])
    
    
