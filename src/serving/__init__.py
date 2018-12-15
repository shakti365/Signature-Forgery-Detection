import os
import markdown
import requests
from flask import Flask, render_template, Markup

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'serving.sqlite')
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/')
    def hello():
        r = requests.get("https://raw.githubusercontent.com/shakti365/Signature-Forgery-Detection/master/One-Shot%20Signature%20Recognition%20Using%20Siamese%20Networks.md")
        content = r.text
        content = Markup(markdown.markdown(content))
        return render_template('index.html', content=content)
    
    @app.route('/howitworks')
    def usage():
        return render_template('usage.html')

    from . import db
    db.init_app(app)

    from . import signature
    app.register_blueprint(signature.bp)

    return app