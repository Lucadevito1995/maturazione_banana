import tensorflow as tf

# Il modulo Image fornisce un'API per creare e manipolare immagini, consentendo di aprire, salvare e modificare immagini
# in vari formati. Ad esempio, è possibile utilizzare il modulo Image per aprire un'immagine, ridimensionarla,
# modificarne il colore o la luminosità e salvarla in un nuovo formato.
# Il modulo ImageOps fornisce una serie di funzioni di elaborazione dell'immagine predefinite per la manipolazione di
# immagini. Ad esempio, il modulo ImageOps offre funzioni come invert (per invertire i colori di un'immagine), mirror
# (per riflettere un'immagine), flip (per ruotare un'immagine) e molte altre funzioni utili.
# Quindi, una volta che si è importato il modulo PIL e i suoi sotto-moduli Image e ImageOps, è possibile utilizzare
# le funzionalità fornite per creare, modificare e salvare immagini.


from PIL import Image, ImageOps
import numpy as np
import streamlit as st

st.write('''
# Banana Ripeness Detection 🍌
''')
st.write("A Image Classification Web App That Detects the Ripeness Stage of Banana")

# pulsante carica immagine
file = st.file_uploader("", type=['jpg', 'png'])


def predict_stage(image_data, model):
    size = (224, 224)

    # ridimensiona image_fata in size,ANTIALIAS è l'algoritmo usato
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)

    # Il secondo frammento di codice np.array(image) crea un nuovo array NumPy a partire dai dati dell'immagine image.
    # L'array NumPy è un tipo di struttura dati multidimensionale utilizzata in Python per elaborare dati numerici.
    # L'array NumPy risultante avrà lo stesso numero di dimensioni dell'immagine originale, ad esempio se l'immagine è in
    # scala di grigi avrà una sola dimensione per i livelli di grigio, mentre se l'immagine è a colori avrà tre dimensioni
    # per i canali di colore rosso, verde e blu.
    # L'array NumPy può quindi essere utilizzato per applicare operazioni matematiche, come ad esempio il calcolo della
    # media, della deviazione standard, la convoluzione con un filtro e molte altre operazioni, utilizzando le funzioni
    # messe a disposizione dal pacchetto NumPy.
    # In sintesi, il frammento di codice importa il pacchetto NumPy e crea un nuovo array NumPy dai dati dell'immagine
    # image, che può essere utilizzato per eseguire operazioni matematiche sulla rappresentazione numerica dell'immagine

    image_array = np.array(image)


    # l'array NumPy image_array viene normalizzato in modo che tutti i suoi valori siano compresi tra -1 e 1.
    # Questo tipo di normalizzazione è spesso utilizzato come pre-elaborazione dei dati per alcuni algoritmi di
    # apprendimento automatico, in quanto permette di migliorare la stabilità numerica del modello e di evitare problemi
    # di saturazione dei valori durante la fase di apprendimento

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1


    # In sintesi, il frammento di codice crea un nuovo array NumPy data per contenere i dati di un'immagine di dimensioni
    # 224x224 con tre canali di colore RGB, utilizzando il tipo di dato np.float32.

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


    # assegna il valore dell'array NumPy normalizzato normalized_image_array alla prima riga dell'array data.
    # In altre parole, il frammento di codice inserisce i dati normalizzati di un'immagine nel contenitore data.
    # La riga [0] indica che si sta assegnando i dati dell'immagine al primo batch del contenitore, mentre
    # normalized_image_array è l'array NumPy contenente i dati dell'immagine normalizzati.
    # L'operazione di assegnazione viene effettuata per ciascuna delle tre dimensioni dell'immagine. Se l'immagine in
    # questione ha dimensioni 224x224 con tre canali di colore RGB, allora l'array normalized_image_array avrà
    # forma (224, 224, 3) e verrà assegnato alla riga [0] dell'array data, in modo che la forma dell'array risultante
    # sia (1, 224, 224, 3).
    # In sintesi, il frammento di codice inserisce i dati normalizzati di un'immagine nell'array data, che può essere
    # utilizzato come input per alcuni algoritmi di apprendimento automatico.

    data[0] = normalized_image_array
    preds = ""


    # Esegue una previsione sull'input data utilizzando il modello di machine learning model.
    # In particolare, il metodo predict esegue la fase di inferenza del modello, ovvero utilizza i pesi appresi dal
    # modello durante la fase di addestramento per produrre un output corrispondente all'input fornito.
    # Il parametro data è l'input per il modello, ovvero l'array NumPy contenente i dati normalizzati dell'immagine di
    # cui si vuole effettuare la previsione. Il modello, a seconda della sua architettura e dei pesi appresi durante la
    # fase di addestramento, produrrà un output corrispondente all'input.
    # Il risultato dell'operazione model.predict(data) sarà un array NumPy contenente la previsione del modello per
    # l'input data. In base al tipo di problema affrontato, la previsione potrebbe essere una singola etichetta di classe
    # o una distribuzione di probabilità su un insieme di classi possibili.
    # In sintesi, il frammento di codice model.predict(data) esegue la previsione di un modello di machine learning
    # sull'input data, restituendo l'output prodotto dal modello.

    prediction = model.predict(data)
    if np.argmax(prediction) == 0:
        st.write(f"ONRIPE")
    elif np.argmax(prediction) == 1:
        st.write(f"OVERRIPE")
    else:
        st.write(f"RIPE")

    return prediction


# Apre un'immagine dal file file utilizzando la libreria Pillow, la visualizza tramite la libreria streamlit, quindi
# carica un modello di machine learning dal file ripeness.h5 e invoca una funzione predict_stage per effettuare una
# previsione sulla classe di maturazione dell'immagine.

# Più in dettaglio:

# La prima riga, image = Image.open(file), utilizza il metodo open della libreria Pillow per aprire l'immagine specificata
# dal percorso file e assegnarla alla variabile image.
# La seconda riga, st.image(image, use_column_width=True), utilizza la libreria streamlit per visualizzare l'immagine
# image. use_column_width=True consente alla visualizzazione dell'immagine di adattarsi alla larghezza della colonna in
# cui è inserita.
# La terza riga,
# model = tf.keras.models.load_model('C:/Users/gri10/banana-ripeness-detection/banana-main/Banana-Ripeness-Detection-main/ripeness.h5'),
# carica un modello di machine learning dal file ripeness.h5 utilizzando la libreria tensorflow.keras.models.
# La quarta riga, Generate_pred = st.button("Predict Ripeness Stage.."), visualizza un pulsante tramite la libreria
# streamlit e assegna il suo stato (premuto o no) alla variabile Generate_pred.
# La quinta riga, if Generate_pred: prediction = predict_stage(image, model), verifica se il pulsante Generate_pred è
# stato premuto, quindi invoca la funzione predict_stage per effettuare una previsione sulla classe di maturazione
# dell'immagine. La previsione viene assegnata alla variabile prediction.
# In sintesi, il frammento di codice permette di caricare un'immagine, visualizzarla e invocare un modello di
# machine learning per effettuare una previsione sulla classe di maturazione dell'immagine.


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    model = tf.keras.models.load_model(
        'C:/Users/gri10/banana-ripeness-detection/banana-main/Banana-Ripeness-Detection-main/ripeness.h5')
    Generate_pred = st.button("Predict Ripeness Stage..")
    if Generate_pred:
        prediction = predict_stage(image, model)

# Il file ripeness.h5 è un file di salvataggio del modello di machine learning utilizzato per effettuare la previsione
# sulla maturazione delle banane.
# In particolare, il file .h5 è un formato di file standard utilizzato per il salvataggio di modelli di machine learning
# in TensorFlow. Esso contiene i pesi del modello, i parametri di configurazione, la topologia della rete neurale,
# nonché altre informazioni necessarie per utilizzare il modello.
# Il file ripeness.h5 in particolare è il risultato dell'addestramento di un modello di deep learning per la '
# classificazione della maturazione delle banane. Esso è stato addestrato su un insieme di immagini di banane di diverse
# classi di maturazione (verdi, gialle e mature) e contiene quindi i pesi e i parametri appresi dal modello durante
# la fase di addestramento.
# In pratica, questo file è utilizzato dal frammento di codice Python per caricare il modello addestrato e utilizzarlo
# per effettuare la previsione sulla classe di maturazione delle banane rappresentate nell'immagine.