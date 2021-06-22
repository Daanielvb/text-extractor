from src.parser.PDFReader import *
from src.parser.DOCReader import *
from src.util.NgramUtil import *
from src.util.GlobalFeatures import *
from sklearn.svm import SVC
from src.classifiers.NeuralNetwork import *
from src.classifiers.RFClassifier import *
from src.classifiers.SimpleNeuralNetwork import *


def extract_text_from_original_works(base_path='../data/students_exercises', raw=False):
    clean_txt_content, txt_file_names, _ = FileUtil.convert_files(base_path, raw)
    clean_doc_content, doc_file_names = DOCReader().convert_docs(base_path, raw)
    clean_pdf_content, pdf_file_names = PDFReader().convert_pdfs(base_path, raw)

    files_content = FileUtil.merge_contents(clean_txt_content, clean_doc_content, clean_pdf_content)
    file_paths = FileUtil.merge_contents(txt_file_names, doc_file_names, pdf_file_names)
    CSVReader.write_files('../data/parsed-data/', file_paths, 'data.csv', files_content)


def extract_text_from_original_works_val(base_path='../../data/varela_dataset', raw=False):
    clean_txt_content, file_paths, folders = FileUtil.convert_files(base_path, raw)
    if not raw:
        clean_txt_content = FileUtil.remove_first_line(clean_txt_content)
    CSVReader.write_files('../data/parsed-data/', file_paths, 'varela-data.csv', clean_txt_content,
                          author_naming=False, columns=['Text', 'Subject', 'Author'], folders=folders)


def prepare_train_data(dataframe):
    Meta_Y = dataframe.pop('Autor')
    Meta_X = dataframe
    X_train, X_test, y_train, y_test = train_test_split(Meta_X, Meta_Y, test_size=0.4, stratify=Meta_Y)
    return X_train, X_test, y_train, y_test


def run_svc(X_train, X_test, y_train, y_test):
    clf = SVC(gamma='scale')
    clf.fit(X_train, y_train)
    print(clf.predict(X_train))
    print(clf.score(X_test, y_test))


def remove_group_works(dataframe):
    dataframe['Author'] = dataframe['Author'].astype('str')
    mask = (dataframe['Author'].str.len() <= 5)
    return dataframe.loc[mask]


def save_plot(plot_name):
    plt.savefig(plot_name)


def show_column_distribution(dataframe, class_name):
    dataframe[class_name].value_counts().plot(kind='barh')
    plt.show()


def save_converted_stylo_data(input_file='../../data/parsed-data/news-dataset.csv', output_file='news-stylo-data.csv'):
    #extract_text_from_original_works()
    CSVReader().write_stylo_features('../../data/parsed-data/', output_file,
                                     CSVReader.transform_text_to_stylo_text(input_file, verbose=False))


def save_converted_stylo_data_binary(idx):
    CSVReader().write_stylo_features('../../data/parsed-data/', 'stylo-data' + str(idx) + '.csv',
                                     CSVReader.transform_text_to_stylo_text('../../data/parsed-data/binary/text/dataset_author_' + str(idx) + '.csv', verbose=False))


def run_compiled_model(model, tokenizer, encoder, X_predict, y_expected):
    embedded_sentence = tokenizer.texts_to_sequences([X_predict])
    padded_sentences = pad_sequences(embedded_sentence, 6658, padding='post')

    pred = model.predict_entries(padded_sentences)

    decoded_pred = ModelUtil.decode_class(encoder, pred[0])
    print('predicted:' + decoded_pred + ' expected:' + y_expected)
    return decoded_pred == y_expected


def run_compiled_pipeline():
    df = pd.read_csv('../../data/parsed-data/data2.csv')
    from random import randint
    # TODO: Check if null tokens is a good approach for unseen content
    # TODO: Perform validation with unseen data
    # TODO: See if its possible to use Random Forest with our embedding matrix
    # TODO: Think about the group works problem
    correct = 0
    df = ModelUtil().remove_entries_based_on_threshold(df, 'Author', 1)
    nn = NeuralNetwork()
    nn.load_model()
    encoder = ModelUtil().load_encoder()
    tokenizer = ModelUtil().load_tokenizer()
    run_compiled_model(nn, tokenizer, encoder,
                       '1 - Competição entre membros de espécies diferentes que usam os mesmos recursos limitados. Os recursos costumam ser limitados em um habitat e muitas espécies podem competir para consegui-los. O efeito geral da competição interespecífica é negativo para as duas espécies participantes, ou seja, cada espécie estaria melhor se a outra espécie não estivesse presente. Com tudo referente à complementaridade de nicho O princípio exclusão competitiva diz que duas espécies competidoras podem concorrer em determinado local, mas para isso elas precisam possuir nichos realizados diferentes. 2 Em referente ao primeiro estudo de caso (Asterionella formosa/ Synedra ulna) a ocorrência competitiva devido ao mesmo recurso (silicato) apresenta princípio da exclusão competitivas onde ambas ocupando o mesmo nicho em que a capacidade suporte influencia na exclusão de uma espécie. No segundo caso a relação de coexistência da diversidade de espécie de peixe-palhaço esta relacionada com a quantidade de anêmonas onde o seu principal recursos esta no abrigo possibilitando a produtividade e perpetuação da espécie de peixe, além disso, devido a população desse peixe esta ligado ao recurso limitante por anêmona criando uma diversidade que utiliza diferentes nichos devido ao distanciamento de cada anêmona do estudo. 3 - O princípio da exclusão competitiva ou, como também é chamado, Lei de Gause, é uma proposição que afirma que, em um ambiente estável no qual os indivíduos se distribuem de forma homogênea, duas espécies com nichos ecológicos parecidos não podem coexistir, devido a pressão evolutiva exercida pela competição. De acordo com esse princípio, um dos competidores terminará por sobrepujar ao outro, o que pode acarretar mudanças morfológicas, comportamentais, deslocamento de nicho ecológico ou até mesmo a extinção da espécie em desvantagem. Em suma, o que esse conceito quer dizer é que competidores completos não podem coexistir. ',
                       'test')

    for i in range(100):
        idx = randint(0, len(df.Author) - 1)
        print('idx:' + str(idx))
        half_size = int(len(df.Text.iloc[idx]) / 2)
        half_text = df.Text.iloc[idx][:half_size]
        pred_result = run_compiled_model(nn, tokenizer, encoder, half_text, df.Author.iloc[idx])
        if pred_result:
            correct += 1

    print('total correct = ' + str(correct))
    print('accuracy % = ' + str((correct / 100) * 100))


def run_stylometric_pipeline(dataset='../../data/parsed-data/news-stylo-data.csv'):
    df = pd.read_csv(dataset)
    #df = ModelUtil().remove_entries_based_on_threshold(df, 'Author', 3)
    nn = SimpleNeuralNetwork(df)
    accs = []
    for i in range(0, 20):
        test, _ = nn.simple_split_train()
        accs.append(test)
    print(np.mean(accs))
    print(np.std(accs))


def run_complete_pipeline(dataset='../../data/parsed-data/data.csv'):
    df = pd.read_csv(dataset)

    df = ModelUtil().remove_entries_based_on_threshold(df, 'Author', 3)
    #show_column_distribution(df, 'Author')

    y = df.pop('Author')

    le = LabelEncoder()
    le.fit(y)
    encoded_Y = le.transform(y)
    ModelUtil().save_encoder(le)
    # decode: le.inverse_transform(encoded_Y)

    tokenizer, padded_sentences, max_sentence_len \
        = PortugueseTextualProcessing().convert_corpus_to_number(df)

    #ModelUtil().save_tokenizer(tokenizer)
    vocab_len = len(tokenizer.word_index) + 1

    glove_embedding = PortugueseTextualProcessing().load_vector(tokenizer)

    embedded_matrix = PortugueseTextualProcessing().build_embedding_matrix(glove_embedding, vocab_len, tokenizer)

    cv_scores = []
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
    models = []

    nn = NeuralNetwork()
    #conv2d = NeuralNetwork()

    nn.build_baseline_model(embedded_matrix, max_sentence_len, vocab_len, len(np_utils.to_categorical(encoded_Y)[0]))

    # TODO: Check an alternative for this validation strategy
    #val_data, X, Y = ModelUtil().extract_validation_data(padded_sentences, encoded_Y)
    y = np_utils.to_categorical(encoded_Y)
    history = nn.model.fit(
        padded_sentences, y,
        validation_split=0.2,
        epochs=250,
        batch_size=512)


    for train_index, test_index in kfold.split(X, Y):
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(Y)
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = dummy_y[train_index], dummy_y[test_index]
        nn.train(X_train, y_train, 100)

        scores = nn.evaluate_model(X_test, y_test)
        cv_scores.append(scores[1] * 100)
        models.append(nn)
    # print("NN accuracy %.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
    # best_model = models[cv_scores.index(max(cv_scores))]
    # #best_model.save_model(model_name='nn.json', weights_name='nn.h5')
    # saved_models.append(best_model)

    # for key, val in val_data.items():
    #     print('key:' + str(key))
    #     for model in saved_models:
    #         # TODO: This is throwing errors in some cases: Evaluate
    #         model.predict_entries(val.reshape(1, 2139))


def clean_and_get_size(df):
    df['Text'] = df['Text'].str.replace('/r', '')
    df['Text'] = df['Text'].str.replace('/n', ' ')
    df['size'] = df['Text'].str.split().str.len()
    # How to retrieve authors information
    #df.groupby('Author')['size'].agg(['sum', 'mean'])
    return df


def build_global_features(df, columns_to_drop=['Text', 'Author']):
    ng = NgramUtil(df['Text'], [1, 3, 4, 5], [10, 3, 3, 3])
    df = GlobalFeatures().calculate_global_hapax_legomena(df)
    df = ng.upgrade_df_with_count(df)
    df.drop(labels=columns_to_drop, axis=1, inplace=True)
    return df


def build_merged_df(df_name):
    df_global = build_global_features(pd.read_csv('../../data/parsed-data/data.csv'))
    df_stylo = pd.read_csv('../../data/parsed-data/stylo-data.csv')
    df_merge = pd.merge(df_stylo, df_global, left_index=True, right_index=True)
    author = df_merge.pop('Author')
    df_merge['Author'] = author
    CSVReader().export_dataframe(df_merge, df_name)
    return df_merge


def build_author_verification_df(df, author_name):
    author_df = df.copy()
    author_df['is_author'] = author_df['Author'].apply(lambda x: int(x == author_name))
    author_df.pop('Author')
    return author_df


def build_authors_verification_dfs(df):
    authors_dict = dict()
    authors = list(set((df['Author'])))
    for author in authors:
        authors_dict[author] = build_author_verification_df(df, author)
    return authors_dict


if __name__ == '__main__':
    # TODO: Check why there are only 86 records instead of 106.

    #save_converted_stylo_data('../../data/parsed-data/varela-data.csv', 'varela-stylo.csv')
    df = pd.read_csv('../data/parsed-data/full-stylo-data.csv')

    results = build_authors_verification_dfs(df)





    #df_stylo = pd.read_csv('../../data/parsed-data/full-stylo-data.csv')

    #df = pd.read_csv('../../data/parsed-data/selected-data.csv')
    #save_converted_stylo_data(input_file='../../data/parsed-data/data.csv', output_file='stylo-data.csv')
    # CSVReader().export_dataframe(df, '../../data/parsed-data/data-without-stopwords.csv')

