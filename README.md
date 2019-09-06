# 1. Create a Python 3.6.1 virtualenv
## a. open cmd, in python project directory enter:
### Command: python -m venv env
## b. activate virtual environment
### Command: env/Scripts/activate
# 2. Install dependencies:
### Command: pip install -r requirements.txt
# 3. Set up postgresql database
## a. Install postgresql:
## b. In django project directory, go to tutorial/settings.py, change DATABASES dictionary to this codes below (remember to change key USER, PASSWORD and NAME of database to your own information):
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'djangorest',
        'USER': 'postgres',
        'PASSWORD': 'loi0342000',
        'HOST': 'localhost',
        'POST': '5432'
    },
    'original': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}
# 4. Run server
### Command: python manage.py runserver
