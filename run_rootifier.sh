for file in Data/*.ch2g; do
    run=$(basename "$file" .ch2g)

    echo "Processing $run"

    build/bin/scripts/100_RootConverter \
        -i "$file" \
        -o "Rootified/intermediate/${run}_RootConverter.root" \
        -n 2

    build/bin/scripts/101_EventRecon \
        -f "Rootified/intermediate/${run}_RootConverter.root" \
        -o "Rootified/intermediate/${run}_EventRecon.root"

    build/bin/scripts/102_EventMatch \
        -f "Rootified/intermediate/${run}_EventRecon.root" \
        -o "Rootified/final/${run}_EventMatch.root"
done