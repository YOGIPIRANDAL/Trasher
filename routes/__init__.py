from .main import main_bp
from .predict import predict_bp
from .predict2 import predict_bp2


# Register blueprints
def register_blueprints(app):
    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(predict_bp2)

# Import blueprints after defining them
from . import main, predict