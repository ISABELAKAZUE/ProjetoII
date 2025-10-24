import os, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
import pandas as pd

df = pd.read_csv('resultados.csv')

paths_imagens = df['file_path']
y = df['Resultados']

path = "stakeholder_images"
label = 'Resultados'

IMG_SIZE = (224, 224)

# Retorna um tensor a partir do caminho da imagem
def load_image(path):
    img = tf.io.read_file(path)    # lê a imagem
    img = tf.image.decode_jpeg(img, channels=3)    # converte a imagem em tensor
    img = tf.image.resize(img, IMG_SIZE, method='bicubic', antialias=True)    # Redimensiona o tensor para o tamanho alvo
    img = tf.keras.applications.mobilenet.preprocess_input(tf.cast(img, tf.float32))    # Normaliza os valores do tensor para [-1,1]
    return img

# Retorna um dataset do Tensorflow
def make_ds(paths, ys, batch_size):
    X = tf.data.Dataset.from_tensor_slices(paths).map(load_image, num_parallel_calls=tf.data.AUTOTUNE)  # Cria um dataset de tensores
    Y = tf.data.Dataset.from_tensor_slices(ys)    # Cria um dataset de rótulos
    ds = tf.data.Dataset.zip((X, Y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)    # Emparelha tensores e rótulos e cria os batches
    return ds

# carrega Backbone e treina Head
def build_model(lr=1e-3):
    base = keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,)) # Carrega o backbone do modelo pré-treinado
    base.trainable = False     # Congela os pesos do backbone
    inp = keras.Input(shape=IMG_SIZE+(3,), name='included_input')    # Cria da entrada da rede
    x = base(inp, training=False)  # Adciona o backbone à entrada e impede que as camadas de BatchNormalization sejam alteradas durante o treino
    x = layers.GlobalAveragePooling2D(name='GAP_Head')(x)   # Adiciona uma camada de média global espacial por canal (substitui Flatten para amostra pequena)
    out = layers.Dense(1, activation='softplus', name='Regressor')(x)    # Adiciona uma camada regressora com uma saída
    model = keras.Model(inp, out)    # Instancia o grafo final
    model.compile(optimizer=keras.optimizers.Adam(lr), 
                  loss='mse', 
                  metrics=['mse','mae'], 
                  jit_compile="auto", 
                  steps_per_execution=16)   # Define otimizador e métrica da regressão
                                           # Otimizador: Adam
                                           # Taxa de aprendizagem: 0.001
                                           # Métrica de treino: mse
                                           # Métrica de validação: mae
    return model, base

# Prepara o modelo para o fine-tuning, descongelando camadas e estipulando a taxa de aprendizagem
def finetune(model, base, lr=1e-5, n_unfreeze=20):
    # Descongela n camadas, da saída para a entrada
    if n_unfreeze > 0:
      for layer in base.layers[-n_unfreeze:]:
          if not isinstance(layer, layers.BatchNormalization):
              layer.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(lr), 
                  loss='mse', metrics=['mse','mae'], 
                  jit_compile="auto", 
                  steps_per_execution=16)

# Integração do Leave One Out Cross-Validation com o ajuste fino do modelo
def fitting_loocv(paths, y, batch_size=8, epochs_head=20, epochs_ft=10, n_unfreeze=1, lr_head=1e-3, lr_ft=1e-5):
    loo = LeaveOneOut()      # O módulo do Scikit Learn se encarrega da iteração entre os folds
    evaluate = []
    y_true_oof = []    # Rótulos reais out-of-fold
    y_pred_oof = []    # Rótulos previstos out-of-fold
    for fold, (tr_idx, te_idx) in enumerate(tqdm(loo.split(paths), total=len(paths), desc="LOOCV"), 1):  # índices do dataset: n-1 índices de treino e 1 índice de teste
        tr_paths, te_paths = np.array(paths)[tr_idx], np.array(paths)[te_idx]    # Encontra os dados pelos índices
        tr_y, te_y = y[tr_idx], y[te_idx]                                        # Encontra os rótulos pelos índices
        ds_tr = make_ds(tr_paths, tr_y, batch_size=batch_size)     # Cria novo DataSet de treino
        ds_te = make_ds(te_paths, te_y, batch_size=1)   # Cria DataSet de teste

        model, base = build_model(lr_head)    # Extrai o modelo pré-treinado com head e input adaptados, e uma backbone imutável
        cb = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]    # Define interrupção quando não houver melhora na validação
        model.fit(ds_tr, epochs=epochs_head, validation_data=ds_te, callbacks=cb, verbose=0)   # Treina somente a head (descongelada)

        # Para salvar csv com as métricas por época
        diretório = f'reports\\fits_csv\\hdep_{epochs_head}\\ftep_{epochs_ft}\\uf_{n_unfreeze}'
        os.makedirs(diretório, exist_ok=True)
        csv = CSVLogger(f'{diretório}\\fold_{fold:03d}ft.csv', separator = ',', append=False)

        finetune(model, base, lr=lr_ft, n_unfreeze=n_unfreeze)    # Descongela camadas e define nova taxa de aprendizagem
        model.fit(ds_tr, epochs=epochs_ft, validation_data=ds_te, callbacks=[cb, csv], verbose=0)    # Treina novamente o modelo

        # Agregando a previsão
        y_pred = float(model.predict(ds_te, verbose = 0).ravel()[0])
        y_pred_oof.append(y_pred)
        y_true_oof.append(te_y)

        # Salva as métricas alvo e de validação
        evaluate.append(model.evaluate(ds_te, verbose=0))

    r2_geral = r2_score(y_true_oof, y_pred_oof)
        
    return r2_geral, evaluate
