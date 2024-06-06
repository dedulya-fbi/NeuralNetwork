# Настройка рабочей среды для нейросети

Шаги установки TensorFlow с помощью pip с официального сайта   [Tensorflow](https://www.tensorflow.org/install/pip?hl=ru). 

Перейдите в __командную строку__ и выполните команду __pip install --upgrade pip__
***

### Устанавливаем Python
Переходим на официальный сайт [Python](https://www.python.org/downloads/windows/) и скачиваем одну из версий __3.9 – 3.11__


### Устанавливаем MiniConda 
[Miniconda](https://docs.anaconda.com/free/miniconda/) — рекомендуемый подход для установки TensorFlow с поддержкой графического процессора. Он создает отдельную среду, чтобы избежать изменения любого установленного программного обеспечения в вашей системе. Это также самый простой способ установить необходимое программное обеспечение, особенно для настройки графического процессора.
***

### Создаем среду conda
Откройте командную строку __Anaconda Prompt__.Создайте новую среду conda с именем __tf__ (или введите свое имя) с помощью следующей команды:

```
conda create --name tf python=3.9
```
Ждем некоторое время, пока создается окружение

Во время создания окружения появится текст: __The following NEW packeges will be INSTALLED__. Жмем __Proceed([y]/n)__ - __y__.
Ждем завершения создания...

Теперь Вы можете деактивировать и активировать окружение с помощью следующих команд:
```
conda deactivate
conda activate tf
```

Давайте активирует __Conda:__
```
conda activate tf
```
После активации окружения __tf__ у Вас изменится путь:
__(base) C:\Users\79524>__ поменяется на __(tf) C:\Users\79524>__

***
#### Настройка графического процессора

Сначала установите [драйвер графического процессора NVIDIA](https://www.nvidia.com/Download/index.aspx), если у вас его еще нет.

Затем установите __CUDA, cuDNN__ с помощью conda.
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```
Процесс установки занимает от пяти до десяти минут
***

#### Установите TensorFlow
Устанавливаем все необходимые пакеты для работы:
```
pip install --upgrade pip
```
Затем установите TensorFlow с помощью __pip__.
```
pip install tensorflow==2.8.0
```
Если вы увидели ошибки, то не обращайте внимания. Мы позже их исправим. Они появляются из-за того, что не все библиотеки установлены.
***

### Установка Jupyter Notebook
Устанавливаем ядро __Jupyter__ следующей командой: 
```
pip install jupyter
```
Устанавливаем сам __Notebook__:
```
pip install notebook
```
Теперь в командной строке мы можете прописать jupyter notebook и запустить его: 
```
(tf) C:\Users\79524>jupyter notebook
```
У Вас откроется веб-интерфейс Jupyter Notebook.Создайте новую пустую папку. Это можно сделать, нажав на кнопку __New__ и в выпадающем списке выбрать __Folder__. 

Мы убедились, что Jupyter Notebook запустился и теперь нам необходимо прописать наше окружение. Откройте окно командной строки, нажмите сочетание клавиш __Ctrl+C__ для завершения работы веб-интерфейс Jupyter Notebook.

Теперь необходимо прописать следующие команды: 
```
pip install ipykernel
```
__ipykernel__ позволяет работать с Jupyter Notebook
```
python -m ipykernel install --user --name myenv --display-name tf
```
Теперь мы может проверить работу нашего окружения.Введем команду __jupyter notebook__ в командной строке  для открытия веб-интерфейс Jupyter. Зайдем в нашу созданную папку. Нажмем на кнопку __New__ и в выпадающем списке найдем наше виртуальное окружения __tf__. Нажимаем на него и нас перебрасывает в интерактивный блокнот Jupyter Notebook.
***

### Установка дополнительных библиотек библиотек
Мы убедились, что Jupyter Notebook запустился и виртуальное окружения заработало. Откройте окно командной строки, нажмите сочетание клавиш Ctrl+C для завершения работы веб-интерфейс Jupyter Notebook.

Установим библиотеку __protobuf__:
```
pip install protobuf==3.19.6
```
В файле __requirements.txt__ находятся список библиотек,необходимых для корректной работы.Чтобы запустить установку, необходимо прописать команду: 
```
pip install -r requirements.txt
```
После установки всех библиотек вы можете проверить работу и импортировать библиотеки.
```
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D,AveragePooling2D,Conv2DTranspose, Input, Concatenate, Add, BatchNormalization, Activation, MultiHeadAttention
import tensorflow_hub as hub
import tensorflow_text as text
from ipywidgets import IntProgress
from IPython.display import display
import cv2
#подключена ли видеокарта?
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
```
После выполнения этого кода Вы можете убедиться, что все библиотеки установлены корректно и Ваша видеокарта подключена. У вас не будет ошибок при выполнении этого кода и выведется информация о подключенной видеокарте:
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

# Программный код и обучение
