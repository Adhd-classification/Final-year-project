from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/dashboard', methods=['POST'])
def login_user():
    username = request.form['username']
    password = request.form['password']
    
    # Check credentials (this is a dummy check, implement real validation)
    if username == 'admin' and password == 'password':
        return render_template('dashboard.html')
    else:
        return 'Invalid credentials', 403

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/create-account')
def create_account():
    return render_template('create_account.html')

@app.route('/register', methods=['POST'])
def register_user():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    # Add registration logic (e.g., save to database)
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Here you can include your code to process the file and generate plots
        # For demonstration purposes, we're just displaying the uploaded file
        return render_template('dashboard.html', images=[filepath])

if __name__ == '__main__':
    app.run(debug=True)
