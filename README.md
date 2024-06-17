<h3>Проект по борьбе с паразитным шумом на изображениях</h3>
_______________________________________________________
**Доступные возможности**

* формирование наборов данных из тензоров Pytorch, списка файлов изображений
* обработка - наложение указанного шума, подавление шума путем обработки автоэнкодером
* визуализация результатов в виде html-страницы

**Структура**

```
denoising_system/
├── app/
│ ├── classification.py
│ ├── dataset.py
│ ├── denoising.py
│ ├── model_autoencoder.py
│ ├── reporter.py
│ └── ...
├── tests/
│ └── ...
└── README.md
```

* classification.py - вычисление результатов классификации
* dataset.py - классы для формирования датасетов
* denoising.py - наложение, подавление шума
* model_autoencoder.py - модель сети автоэнкодера для подавления шума
  на [Pytorch](https://pytorch.org/docs/stable/index.html)
* reporter.py - генерация отчета по результатам классификации

**Примеры использования**

1. Создание объекта `AdvImgDataset` для работы с файлами изображений, например, изображениями без шума:

```python
# создаем датасет из изображений
# dir_path - путь до папки с изображениями
# PREPROCESSING_TRANSFORM - способ трансформирования изображений при необходимости
clean_dataset = AdvImgDataset(label_list=[0, 1, 2],
                              img_list=[dir_path + 'Person', dir_path + 'Car', dir_path + 'Boat'],
                              transform=PREPROCESSING_TRANSFORM)
```

2. Создание модели автоэнкодера для шумоподавления:

```python
denoising_model = AutoencoderModel(is_custom_pretrained=False).autoencoder
```

3. Визуализация результатов классификации до и после шумоподавления:

```python
# classification_model - модель классификации
# true_labels - корректные классы
# clean_data_tensor - данные изображений без шума в виде тензоров
# adv_data_tensor - данные изображений с шумом в виде тензоров
# denoising_data_tensor - данные изображений после шумоподавления в виде тензоров
acc_list = calculate_adv_prediction_multi_with_dump(classification_model,
                                                    true_labels,
                                                    {"clean": clean_data_tensor,
                                                     "adv": adv_data_tensor,
                                                     "denoising": denoising_data_tensor})
for acc, case_name in zip(acc_list, ["clean", "adv", "denoising"]):
    print(f'Accuracy {case_name}: {acc}')
# запуск генерации отчета с результатами
reporter()
```

Больше примеров в [main.py](./app/main.py)