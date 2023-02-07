cd C:\Users\User\Desktop\kiel\teaching\seminars\datascience_in_practice_psyM1_2
cd ..
jupyter-book build datascience_in_practice_psyM1_2
cd datascience_in_practice_psyM1_2
ghp-import -n -p -f _build/html
git add .
