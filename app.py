from flask import Flask, render_template, request
from search import SearchEngine

engine = None

def load_engine():
    global engine
    if engine is None:
        print("Loading search engine...")
        engine = SearchEngine("./out_index")
        print("Search engine ready!")
    return engine

app = Flask(__name__)

@app.route('/')
def home():
    load_engine()
    return render_template('index.html')

@app.route('/search')
def search():
    engine = load_engine()

    query = request.args.get('q', '').strip()
    if not query:
        return render_template('index.html', error="Please enter a search query")

    results, search_time = engine.search(query, top_k=10)
    return render_template(
        'results.html',
        query=query,
        results=results,
        search_time=search_time,
        num_results=len(results)
    )

if __name__ == '__main__':
    app.run(debug=False, port=5000)
