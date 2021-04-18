## README.txt
#example run 
python3 ./determineCarPrices.py ./vehicles_medium.csv -v -hu "[20, 20]" -out "[price]" -in "[year, manufacturer, model, condition, cylinders, fuel, odometer, title_status, transmission, drive, paint_color]" -u "[2010, 1, 1, 1, 1, 1, 43000, 1, 1, 1, 1]"
python3 ./determineCarPrices.py ./vehicles_medium.csv -out "[price]" -in "[year, manufacturer, model, condition, cylinders, fuel, odometer, title_status, transmission, drive, paint_color]" -u "[2014, 2, 2, 2, 2, 1, 93000, 1, 2, 2, 0]"
python3 ./determineCarPrices.py ./vehicles_small.csv -v -hu "[20, 20]" -out "[price]" -in "[year]"

## New changes
python3 ./determineCarPrices.py -train_csv ./vehicles_huge.csv -hu "[50, 50, 50, 50, 50, 50]" -out "[price]" -in "[year, manufacturer, model, condition, cylinders, fuel, odometer, title_status, transmission, drive, paint_color]"
python3 ./determineCarPrices.py -u "[2010, 1, 1, 1, 1, 1, 43000, 1, 1, 1, 1]"