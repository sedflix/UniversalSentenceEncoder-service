from flask import Flask, render_template, request
import numpy as np
from UniversalSentenceEncoder import UniversalSentenceEncoder


app = Flask(__name__)

obj = ""

@app.route('/api/similarity')
def api_from_id_json():
    json_data = request.get_json()

    try:
        text_1 = str(json_data['text2'])
        text_2 = str(json_data['text2'])
    except:
        return jsonify({
            "msg": "Wrong parameters."
        }), 403

    features = obj.get_sent_vector(text_1, text_1])
    ans = np.inner(features, features)
    return str(ans[0][1])



if __name__ == '__main__':
    obj = UniversalSentenceEncoder()
    app.run(host='0.0.0.0', port=5000, debug=True)
