import nltk
import networkx as nx
import matplotlib.pyplot as plt


def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)


def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences


def extract_names(document):
    names = []
    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    names.append(' '.join([c[0] for c in chunk]))
    return names


def extract_organizations(document):
    organizations = []
    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'ORGANIZATION':
                    organizations.append(' '.join([c[0] for c in chunk]))
    return organizations


def draw_graph(graph, label):
    # create directed networkx graph
    G = nx.Graph()

    # add edges
    G.add_edges_from(graph)
    # graph_pos = nx.shell_layout(G)
    # graph_pos = nx.spectral_layout(G)
    graph_pos = nx.spring_layout(G)
    # graph_pos = nx.random_layout(G)

    # draw nodes, edges and labels
    nx.draw_networkx_nodes(G, graph_pos, node_size=1000, node_color='blue', alpha=0.3)

    # we can now added edge thickness and edge color
    nx.draw_networkx_edges(G, graph_pos, width=2, alpha=0.3, edge_color='green')
    nx.draw_networkx_labels(G, graph_pos, font_size=12, font_family='sans-serif')

    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=label)

    plt.show()


def main():
    entity = []
    for i in range(1, 11):
        file = open("artikel" + str(i) + ".txt", "r")
        string = remove_non_ascii(file.read())
        names = extract_names(string)
        names = list(set(names))
        entity.append(names)

    graph = []
    label = {}
    temp = {}
    for item in entity:
        for i in item:
            for j in item:
                if i != j:
                    val = (i, j)
                    if temp.has_key(val):
                        temp[val] = temp[val] + 1
                        if temp[val] > 1:
                            label[val] = temp[val]
                            graph.append(val)
                    else:
                        temp[val] = 1

    draw_graph(graph, label)

main()
