import math

def scalar_div_list(list_to_div, scalar):
    return [math.ceil(x/scalar) for x in list_to_div]

def variant_size_setup(variant, size):
    # Variant 1
    if variant == "variant1":
        # size=1 (x0.5)
        if size == 1:
            nb_pts = [20, 30, 40]
            nb_test_pts = scalar_div_list(nb_pts, 4)
            nb_valid_pts = scalar_div_list(nb_pts, 4)

        # size=2 (x0.75)
        elif size == 2:
            nb_pts = [30, 45, 60]
            nb_test_pts = scalar_div_list(nb_pts, 4)
            nb_valid_pts = scalar_div_list(nb_pts, 4)

        # size=3 (x1)
        elif size == 3:
            nb_pts = [40, 60, 80]
            nb_test_pts = scalar_div_list(nb_pts, 4)
            nb_valid_pts = scalar_div_list(nb_pts, 4)

        # size=4 (x1.25)
        elif size == 4:
            nb_pts = [50, 75, 100]
            nb_test_pts = scalar_div_list(nb_pts, 4)
            nb_valid_pts = scalar_div_list(nb_pts, 4)

    # Variant 2 or 3 or 4
    elif variant == "variant2" or variant == "variant3" or variant == "variant4":
        # size=1 (x0.5)
        if size == 1:
            nb_pts = [50, 60, 70]
            nb_test_pts = scalar_div_list(nb_pts, 4)
            nb_valid_pts = scalar_div_list(nb_pts, 4)

        # size=2 (x0.75)
        elif size == 2:
            nb_pts = [75, 90, 105]
            nb_test_pts = scalar_div_list(nb_pts, 4)
            nb_valid_pts = scalar_div_list(nb_pts, 4)

        # size=3 (x1)
        elif size == 3:
            nb_pts = [100, 120, 140]
            nb_test_pts = scalar_div_list(nb_pts, 4)
            nb_valid_pts = scalar_div_list(nb_pts, 4)

        # size=4 (x1.25)
        elif size == 4:
            nb_pts = [125, 150, 175]
            nb_test_pts = scalar_div_list(nb_pts, 4)
            nb_valid_pts = scalar_div_list(nb_pts, 4)

    # Variant 5
    elif variant == "variant5":
        # size=1 (x0.5)
        if size == 1:
            nb_pts = [60, 70, 80]
            nb_test_pts = scalar_div_list(nb_pts, 4)
            nb_valid_pts = scalar_div_list(nb_pts, 4)

        # size=2 (x0.75)
        elif size == 2:
            nb_pts = [90, 105, 120]
            nb_test_pts = scalar_div_list(nb_pts, 4)
            nb_valid_pts = scalar_div_list(nb_pts, 4)

        # size=3 (x1)
        elif size == 3:
            nb_pts = [120, 140, 160]
            nb_test_pts = scalar_div_list(nb_pts, 4)
            nb_valid_pts = scalar_div_list(nb_pts, 4)

        # size=4 (x1.25)
        elif size == 4:
            nb_pts = [150, 175, 200]
            nb_test_pts = scalar_div_list(nb_pts, 4)
            nb_valid_pts = scalar_div_list(nb_pts, 4)


    return nb_pts, nb_test_pts, nb_valid_pts