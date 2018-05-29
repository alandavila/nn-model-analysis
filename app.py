from flask import (
    Flask,
    render_template,
    request,
    redirect,
    jsonify)

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

# create route that renders index.html template
@app.route("/")
def home():
    """Return the dashboard homepage."""
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False)
