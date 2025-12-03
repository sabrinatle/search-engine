from flask import Flask, render_template, request
from search import SearchEngine

app = Flask(__name__)

# Load search engine once at startup
print("Loading search engine...")
engine = SearchEngine("./out_index")
print("Search engine ready!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()

    if not query:
        return render_template('index.html', error="Please enter a search query")

    # Perform search
    results, search_time = engine.search(query, top_k=10)

    return render_template('results.html',
                         query=query,
                         results=results,
                         search_time=search_time,
                         num_results=len(results))

if __name__ == '__main__':
    app.run(debug=True, port=5000)