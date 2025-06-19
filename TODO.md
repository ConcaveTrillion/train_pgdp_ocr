Add a button to join a line with next line across to right/left
Add button to refresh page images
Add button to refresh all line images
Add button to do 'crop bottom' and 'crop top' and 'crop both' in edit box

Add button to 'rematch ground truth to current OCR'

Add way to tag italic and small caps text (for future classification?)

Interesting pages which need more OCR processing work to properly match:

From Magic to Science: 69, 70, 71, 72, 73, 74, 99

77 - has fractions. Add the unicode fraction slash see if it can handle it
105 - image with very small caption text with italics
108 - Very complex table/timeline layout


History of the american people:
0 - the final blackletter line won't split properly
50 - line order makes matching not work

The book of Filial Duty:
4 - table of contents
5, 6 - dropcap

Credulities Past & Present
14, 21 - italic dropcap
21 - 3 blockquotes with quotation marks



Things to do:

Blockquote paragraphs
    sometimes quotation mark means first line is slightly more left justified

Recognizing Drop-Caps - train model to recognize the drop cap as a separate word, but join it up with next word in post processing?

Split doesn't always work correctly, sometimes an extra ground truth word is created

laying out pages:

left side notes:
column that takes up <30% of page

right side notes:
column that takes up <30% of page

—