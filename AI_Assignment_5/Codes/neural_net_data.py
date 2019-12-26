#
# Training and Test Data used in neural_net_tester.py
#

"""
1++
0-+
 01
"""
or_data = ((0,0,0),
           (0,1,1),
           (1,0,1),
           (1,1,1),
           (0.25,0,0),
           (0,0.25,0))

or_test_data = ((0.1,0.1,0),
                (0.1,0.9,1),
                (0.9,0.1,1),
                (0.9,0.9,1))
"""
1-+
0--
 01
"""
and_data = ((0,0,0),
            (0,1,0),
            (1,0,0),
            (1,1,1),
            (0.75,1.0,1),
            (1.0,0.75,1))

and_test_data = ((0.1,0.1,0),
                 (0.1,0.9,0),
                 (0.9,0.1,0),
                 (0.9,0.9,1))

"""
1-+
0+-
 01
"""
equal_data = ((0,0,1),
              (0,1,0),
              (1,0,0),
              (1,1,1))

equal_test_data = ((0.1,0.1,1),
                   (0.1,0.9,0),
                   (0.9,0.1,0),
                   (0.9,0.9,1))

"""
1+-
0-+
 01
"""
neq_data = ((0,0,0),
            (0,1,1),
            (1,0,1),
            (1,1,0))

neq_test_data = ((0.1,0.1,0),
                 (0.1,0.9,1),
                 (0.9,0.1,1),
                 (0.9,0.9,0))


"""
3-++-
2-++-
1-++-
0-++-
 0123
"""
vert_band_data = ((0,0,0),
                  (0,1,0),
                  (0,2,0),
                  (0,3,0),
                  (1,0,1),
                  (1,1,1),
                  (1,2,1),
                  (1,3,1),
                  (2,0,1),
                  (2,1,1),
                  (2,2,1),
                  (2,3,1),
                  (3,0,0),
                  (3,1,0),
                  (3,2,0),
                  (3,3,0))

vert_band_test_data = ((0,    1, 0),
                        (0,    2, 0),
                        (0,  1.5, 0),

                        (1.5,  2, 1),
                        (1.5,  5, 1),
                        (1.5,  1, 1),

                        (3,    1, 0),
                        (3,  1.5, 0),
                        (3,    2, 0),

                        (1,  1.5, 1),
                        (1, -1.5, 1),
                        (2,  1.5, 1),
                        (2, -1.5, 1),

                        (4,  0,   0),
                        (4,  4,   0),
                        (-1, 0,   0),
                        (-1, 4,   0))

"""
3----
2++++
1++++
0----
 0123
"""
horiz_band_data = ((0,0,0),
                   (0,1,1),
                   (0,2,1),
                   (0,3,0),
                   (1,0,0),
                   (1,1,1),
                   (1,2,1),
                   (1,3,0),
                   (2,0,0),
                   (2,1,1),
                   (2,2,1),
                   (2,3,0),
                   (3,0,0),
                   (3,1,1),
                   (3,2,1),
                   (3,3,0))

horiz_band_test_data = ((1, 1.5, 1),
                        (2, 1.5, 1),
                        (3, 1.5, 1),
                        (0, 1.5, 1),
                        (4,   0, 0),
                        (4,   4, 0),
                        (-1,  0, 0),
                        (-1,  4, 0))

"""
4--- +
3-- +
2- + -
1 + --
0+ ---
 01234
"""
diag_band_data = ((0,0,1),
                  (1,1,1),
                  (2,2,1),
                  (3,3,1),
                  (4,4,1),
                  (0,4,0),
                  (4,0,0),
                  (0,3,0),
                  (3,0,0),
                  (0,2,0),
                  (2,0,0),
                  (1,4,0),
                  (4,1,0),
                  (1,3,0),
                  (3,1,0),
                  (2,4,0),
                  (4,2,0),
                  )

diag_band_test_data = ((-1,-1,1),
                       (5,  5,1),
                       (-2,-2,1),
                       (6,  6,1),
                       (3.5,3.5,1),
                       (1.5,1.5,1),
                       (4,  0,0),
                       (0,  4,0))

"""
4+++ -
3++ -
2+ - +
1 - ++
0- +++
 01234
"""
idiag_band_data = ((0,0,0),
                   (1,1,0),
                   (2,2,0),
                   (3,3,0),
                   (4,4,0),
                   (0,4,1),
                   (4,0,1),
                   (0,3,1),
                   (3,0,1),
                   (0,2,1),
                   (2,0,1),
                   (1,4,1),
                   (4,1,1),
                   (1,3,1),
                   (3,1,1),
                   (2,4,1),
                   (4,2,1),
                   )

idiag_band_test_data = ((-1,-1,0),
                        (5,  5,0),
                        (-2,-2,0),
                        (6,  6,0),
                        (3.5,3.5,0),
                        (1.5,1.5,0),
                        (4,  0,1),
                        (0,  4,1))


"""
4-----
3-   -
2- + -
1-   -
0-----
 01234
"""
moat_data = ((0,0,0),
             (1,0,0),
             (2,0,0),
             (3,0,0),
             (4,0,0),

             (1,1,0),
             (4,1,0),

             (1,2,0),
             (3,3,1),
             (4,2,0),

             (1,4,0),
             (4,4,0),

             (0,4,0),
             (1,4,0),
             (2,4,0),
             (3,4,0),
             (4,4,0),
             )

moat_test_data = moat_data

"""
4+-
3+-
2+-
1+----
0-++++
 01234
"""
letter_l_data = ((0,0,0),
                 (1,0,1),
                 (2,0,1),
                 (3,0,1),
                 (4,0,1),

                 (1,1,0),
                 (2,1,0),
                 (3,1,0),
                 (4,1,0),

                 (0,2,1),
                 (1,2,0),

                 (0,3,1),
                 (1,3,0),

                 (0,4,1),
                 (1,4,0),
                 )

letter_l_test_data = letter_l_data

def load_csv(filename):
  df = pd.read_csv(filename)
  data = []
  for _, row in df.iterrows():
    row = row.tolist()
    data.append(row)
  return data

two_moons_data = load_csv('two-moons/train.csv')
two_moons_test_data = load_csv('two-moons/test.csv')


simple_data_sets = [("OR", or_data, or_test_data),
                    ("AND", and_data, and_test_data)
                    ]

harder_data_sets = [("EQUAL", equal_data, equal_test_data),
                    ("NOT_EQUAL", neq_data, neq_test_data),
                    ("horizontal-bands", horiz_band_data, horiz_band_test_data),
                    ("vertical-bands", vert_band_data, vert_band_test_data),
                    ("diagonal-band", diag_band_data, diag_band_test_data),
                    ("inverse-diagonal-band", idiag_band_data,
                     idiag_band_test_data)
                    ]

challenging_data_sets = [("moat", moat_data, moat_test_data),
                         ("letter-l", letter_l_data, letter_l_test_data),
                         ]

two_moons_data_set = [("two-moons", two_moons_data, two_moons_test_data)]

all_data_sets = simple_data_sets + harder_data_sets + challenging_data_sets + \
                two_moons_data_set
