# floyd run --env pytorch:py2 --data rNdSANu8G6nFixvQ6Huqnf --gpu "sh floyd_lstm12345.sh"
python sentiment.py --name lstm_sorted_glove_1 --model lstm --logs /output --save /output --embedding glove --optim adagrad --lr 0.05 --wd 1e-4 --emblr 0.1 --data /input/ --glove /media/vdvinh/25A1FEDE380BDADA/ff/glove_sorted &
python sentiment.py --name lstm_sorted_glove_2 --model lstm --logs /output --save /output --embedding glove --optim adagrad --lr 0.05 --wd 1e-4 --emblr 0.1 --data /input/ --glove /media/vdvinh/25A1FEDE380BDADA/ff/glove_sorted &
python sentiment.py --name lstm_sorted_glove_3 --model lstm --logs /output --save /output --embedding glove --optim adagrad --lr 0.05 --wd 1e-4 --emblr 0.1 --data /input/ --glove /media/vdvinh/25A1FEDE380BDADA/ff/glove_sorted &
python sentiment.py --name lstm_sorted_glove_4 --model lstm --logs /output --save /output --embedding glove --optim adagrad --lr 0.05 --wd 1e-4 --emblr 0.1 --data /input/ --glove /media/vdvinh/25A1FEDE380BDADA/ff/glove_sorted &
python sentiment.py --name lstm_sorted_glove_5 --model lstm --logs /output --save /output --embedding glove --optim adagrad --lr 0.05 --wd 1e-4 --emblr 0.1 --data /input/ --glove /media/vdvinh/25A1FEDE380BDADA/ff/glove_sorted &
wait
echo 'meow'