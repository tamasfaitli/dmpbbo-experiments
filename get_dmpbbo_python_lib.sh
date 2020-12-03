
echo "Cloning https://github.com/tamasfaitli/dmpbbo.git"
git clone --single-branch --branch phase_stopping https://github.com/tamasfaitli/dmpbbo.git

echo "Parsing only python project from library"
mv dmpbbo/dmpbbo_lib dmpbbo_lib

echo "Removing not needed library files"
rm -rf dmpbbo

echo "Ready"
