from datetime import datetime
from time import sleep

while True:
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f'Script running at: {now}')
    sleep(1)
