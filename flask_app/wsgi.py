"""
wsgi.py - Entry point untuk production server (Gunicorn, uWSGI, dll.)
======================================================================
Pastikan load_data() dipanggil sebelum request pertama diterima.
"""
from flask_app.app import app, load_data

# Load semua data saat modul diimport oleh WSGI server
load_data()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
