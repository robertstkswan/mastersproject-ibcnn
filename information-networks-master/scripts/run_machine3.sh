(
(python3.6 main_mi.py -e=1500 -ts=0.05 -af=relu -mie="bins" -bs=512 -ns="10,8,6,4" -d=0.5;
python3.6 main_mi.py -e=1500 -ts=0.10 -af=relu -mie="bins" -bs=512 -ns="10,8,6,4" -d=0.5;
python3.6 main_mi.py -e=1500 -ts=0.15 -af=relu -mie="bins" -bs=512 -ns="10,8,6,4" -d=0.5) &
(python3.6 main_mi.py -e=1500 -ts=0.20 -af=relu -mie="bins" -bs=512 -ns="10,8,6,4" -d=0.5;
python3.6 main_mi.py -e=1500 -ts=0.25 -af=relu -mie="bins" -bs=512 -ns="10,8,6,4" -d=0.5;
python3.6 main_mi.py -e=1500 -ts=0.30 -af=relu -mie="bins" -bs=512 -ns="10,8,6,4" -d=0.5) &
(python3.6 main_mi.py -e=1500 -ts=0.35 -af=relu -mie="bins" -bs=512 -ns="10,8,6,4" -d=0.5) &
(python3.6 main_mi.py -e=1500 -ts=0.40 -af=relu -mie="bins" -bs=512 -ns="10,8,6,4" -d=0.5;
python3.6 main_mi.py -e=1500 -ts=0.45 -af=relu -mie="bins" -bs=512 -ns="10,8,6,4" -d=0.5;
python3.6 main_mi.py -e=1500 -ts=0.50 -af=relu -mie="bins" -bs=512 -ns="10,8,6,4" -d=0.5)
);
echo "Subject: machine3

 relu batch finished" | ssmtp ag939@cam.ac.uk
