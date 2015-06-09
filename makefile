# Coursera, Regression Models, Final Project
#
# exploratory analysis of the mtcars data set
#

# tmuxinator an R dev environment
create_env:
	tmuxinator start r-sandbox

render:
	./R/rmdToHtml.R ml_project

gh-pages_create:
	git branch gh-pages
	git checkout gh-pages
	git push origin gh-pages
	touch .nojekyll
	git add .nojekyll
	git commit -a -m "added a .nojekyll file"
	git push origin gh-pages




# remove generated files
clean:
	rm -f $(SRC).csv
	rm -f *.html *.md
	rm -rf figure/

