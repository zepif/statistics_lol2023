from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import pandas as pd
import os

app = Flask(__name__, static_folder='templates/static')
app.secret_key = 'your_secret_key'

csv_folder = 'D:/statistics/templates/csv_files'
png_folder = 'D:/statistics/templates/png_files'

if not os.path.exists(png_folder):
    os.makedirs(png_folder)

@app.route('/')
def file_list():
    csv_files = [filename for filename in os.listdir(csv_folder) if filename.endswith('.csv')]

    data = {}
    
    for file_name in csv_files:
        file_path = os.path.join(csv_folder, file_name)
        df = pd.read_csv(file_path, error_bad_lines=False)
        data[file_name] = df
    
    png_files = [filename for filename in os.listdir(png_folder) if filename.endswith('.png')]

    return render_template('file_list.html', csv_files=csv_files, data=data, png_files=png_files)

@app.route('/png_files/<filename>')
def display_png(filename):
    return send_from_directory(png_folder, filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'png_file' not in request.files:
        return redirect(request.url)

    png_file = request.files['png_file']

    if png_file.filename == '':
        return redirect(request.url)

    if not png_file.filename.endswith('.png'):
        return redirect(request.url)

    png_file.save(os.path.join(png_folder, png_file.filename))

    return redirect(url_for('file_list'))

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'csv_file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)

    
    csv_file = request.files['csv_file']

    if csv_file.filename == '':
        return redirect(request.url)

    if csv_file:
        csv_file.save(os.path.join(csv_folder, csv_file.filename))
        flash('CSV file uploaded successfully', 'success')

    return redirect('/')



if __name__ == '__main__':
    app.run(debug=True)