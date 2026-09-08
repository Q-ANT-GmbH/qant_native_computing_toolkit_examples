from IPython.display import HTML, display


def display_result(categories, certainties):
    display(
        HTML(
            f'<p style="font-size: 48px;">The provided image contains: <b>{categories[0]}</b>'
        )
    )
    display(
        HTML(
            f'<p style="font-size: 28px;">with a certainty of: {certainties[0] * 100:.1f}%</p>'
        )
    )
    display(HTML('<p style="font-size: 24px;">Less likely options are:</p>'))
    for i in range(1, 5):
        display(
            HTML(
                f'<p style="font-size: 20px;">Less likely options: {categories[i]} with {certainties[i] * 100:.1f}%</p>'
            )
        )
    display(
        HTML(
            '<p style="font-size: 24px;">Q.ANT\'s Photonic AI promises 30 times less power consumption than electronics!</p>'
        )
    )
