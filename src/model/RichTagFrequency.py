

class RichTagFrequency:

    def __init__(self, type, tagged_text):
        self.type = type
        self.tagged_text = tagged_text
        self.size = len(set(self.tagged_text))
        self.count = self.build_counting()

    def build_counting(self):
        count = dict()
        for i in list(set(self.tagged_text)):
            count[i] = self.tagged_text.count(i)
        return count

    def get_male(self):
        return

    def female_male(self):
        return


class RichTags:

    def __init__(self, tags, size):
        self.tags = tags
        self.size = size

    def get_tag_freq_by_type(self, tag_type, kind):
        tags = [tag.count for tag in self.tags if tag.type == tag_type]
        if len(tags) > 0 and kind in tags[0]:
            return tags[0][kind] / self.size
        else:
            return 0

    def get_male(self):
        return self.get_tag_freq_by_type('Gender', 'Masc')

    def get_female(self):
        return self.get_tag_freq_by_type('Gender', 'Fem')

    def get_unspecified_gender(self):
        return self.get_tag_freq_by_type('Gender', 'Unsp')

    def get_singular(self):
        return self.get_tag_freq_by_type('Number', 'Sing')

    def get_plural(self):
        return self.get_tag_freq_by_type('Number', 'Plur')

    def get_participle_verbs(self):
        return self.get_tag_freq_by_type('VerbForm', 'Part')

    def get_infinitive_verbs(self):
        return self.get_tag_freq_by_type('VerbForm', 'Inf')

    def get_finitive_verbs(self):
        return self.get_tag_freq_by_type('VerbForm', 'Fin')

    def get_gerund_verbs(self):
        return self.get_tag_freq_by_type('VerbForm', 'Ger')

    def get_first_person(self):
        return self.get_tag_freq_by_type('Person', '1')

    def get_third_person(self):
        return self.get_tag_freq_by_type('Person', '3')

    def get_past_tense(self):
        return self.get_tag_freq_by_type('Tense', 'Past')

    def get_future_tense(self):
        return self.get_tag_freq_by_type('Tense', 'Fut')

    def get_present_tense(self):
        return self.get_tag_freq_by_type('Tense', 'Pres')

    def get_almost_perfect_past(self):
        return self.get_tag_freq_by_type('Tense', 'Pqp')

    def get_almost_perfect_past(self):
        return self.get_tag_freq_by_type('Tense', 'Imp')

    def get_pron_type_relative(self):
        return self.get_tag_freq_by_type('PronType', 'Imp')

    def get_pron_type_relative(self):
        return self.get_tag_freq_by_type('PronType', 'Rel')

    def get_pron_type_indicative(self):
        return self.get_tag_freq_by_type('PronType', 'Ind')

    def get_pron_type_demons(self):
        # TODO Implement missing types: Emp, Prs, Art, Tot PronType
        return self.get_tag_freq_by_type('PronType', 'Dem')








