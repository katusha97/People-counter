### План до 20 октября
- [ ] В репозитории должна появится программа, которая зачитывает видео (путь к видео передается через аргументы командной строки). 

- [ ] На первых n кадров в середине кадра рисует квадрат и сохраняет кадры с квадратами в папку (путь тоже передает через аргументы командой строки, и n через них). 

Для работы с изображениями используйте opencv, для аргументов командой строки argparse.

- [ ] Main part(object tracking)
- [ ] Readme
- [ ] setup.py
- [ ] command line args
- [ ] Тесты → git action
- [ ] flake8 → git action
- [ ] logging

Запуск скрипта с обучающими картинками:
```
python people_counter.py
--model
/path/to/People-Counter/saved_model/saved_model
--input
/path/to/People-Counter/frames
--skip-frames
1
--confidence
0.2
--output
/path/to/People-Counter/result.mp4
```

Запуск скрипта с тестовым видео:
```
python people_counter.py
--model
/path/to/People-Counter/saved_model/saved_model
--input_video
/path/to/People-Counter/ACAM3.mp4
--skip-frames
10
--confidence
0.3
--output
/path/to/People-Counter/result.mp4
```
