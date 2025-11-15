import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import os
from werkzeug.utils import secure_filename
from utlits import predict_image

app = Flask(__name__)
app.secret_key = 'unmyeong'

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DATABASE = 'database.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            phone_number TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            predicted_class TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone_number = request.form['phone_number']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        conn = get_db_connection()
        user_check = conn.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?',
            (username, email)
        ).fetchone()

        if user_check:
            flash('Username or email already exists. Please choose a different one.', 'error')
            conn.close()
            return render_template('register.html')
        hashed_password = generate_password_hash(password)

        try:
            conn.execute(
                'INSERT INTO users (username, email, phone_number, password) VALUES (?, ?, ?, ?)',
                (username, email, phone_number, hashed_password)
            )
            conn.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('An error occurred during registration. Please try again.', 'error')
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/home')
def home():
    if 'username' not in session:
        flash('Please log in to access the home page.', 'error')
        return redirect(url_for('login'))

    conn = get_db_connection()
    user = conn.execute(
        'SELECT username, email, phone_number FROM users WHERE username = ?',
        (session['username'],)
    ).fetchone()
    conn.close()

    return render_template('home.html', user=user)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        flash("You must log in to make a prediction.", "error")
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'image' not in request.files:
            flash("No file uploaded.", "error")
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash("No selected file.", "error")
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run prediction
            predicted_class = predict_image(filepath, model_path="model/hybrid_cnn_vit.pth")

            # Get user id from DB
            conn = get_db_connection()
            user = conn.execute('SELECT * FROM users WHERE username = ?', (session['username'],)).fetchone()

            # Save prediction
            conn.execute(
                'INSERT INTO predictions (user_id, predicted_class) VALUES (?, ?)',
                (user['id'], predicted_class)
            )
            conn.commit()
            conn.close()

            flash(f"Prediction complete: {predicted_class}", "success")
            return render_template("predict.html", result=predicted_class)

    return render_template("predict.html")

@app.route('/history')
def history():
    if 'username' not in session:
        flash("You must log in to view history.", "error")
        return redirect(url_for('login'))

    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (session['username'],)).fetchone()

    predictions = conn.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC',
        (user['id'],)
    ).fetchall()
    conn.close()

    return render_template('history.html', predictions=predictions)


@app.route('/analytics')
def analytics():
    if 'username' not in session:
        flash("You must log in to view analytics.", "error")
        return redirect(url_for('login'))

    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (session['username'],)).fetchone()

    results = conn.execute(
        'SELECT predicted_class, COUNT(*) as count FROM predictions WHERE user_id = ? GROUP BY predicted_class',
        (user['id'],)
    ).fetchall()
    conn.close()

    labels = [row['predicted_class'] for row in results]
    counts = [row['count'] for row in results]

    return render_template('analytics.html', labels=labels, counts=counts)


@app.route('/datascience')
def datascience():
    return render_template('datascience.html')

@app.route('/exsisting')
def exsisting():
    return render_template('exsisting.html')

@app.route('/proposed')
def proposed():
    return render_template('proposed.html')

if __name__ == '__main__':
    app.run(debug=True)
