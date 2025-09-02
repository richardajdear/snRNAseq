# Code for aggregating and plotting hex bins

library(sf)

make_cells_sf <- function(scores) {
    # Convert scores to surface object with age and pseudotime coordinates
    cells_sf <- scores %>% 
        mutate(Pseudotime_match_dim = (Pseudotime-40)/60*5) %>% 
        st_as_sf(coords = c('Age_log2', 'Pseudotime_match_dim'))
}

define_hex_grid <- function(cells_sf) {
    # Define the hex grid in the coordinate space of the cells
    hex_grid <- cells_sf %>% 
        st_make_grid(n = c(7,7), square=FALSE) %>% 
        st_as_sf() %>% mutate(hex_id = row_number())

    return(hex_grid)
}

aggregate_cells_by_hex <- function(cells_sf, hex_grid) {
    # Match cells to their hex and aggregate
    hex_cells <- cells_sf %>% st_join(hex_grid) %>% st_drop_geometry() %>% 
        group_by(TF, hex_id) %>% 
        summarize(count=n(), score = mean(score))

    return(hex_cells)
}

plot_hexes <- function(hex_cells, hex_grid, TFs_to_plot = c('ETS2', 'NFKB1')) {

    # Test code for plotting cells on the hexes
    # points_df <- data.frame(st_coordinates(cells_sf)) %>% 
    # mutate(TF=cells_sf$TF, score=cells_sf$score) %>% 
    # filter(TF%in%TFs_to_plot) %>% 
    # group_by(TF) %>% 
    # mutate(score = (score-mean(score))/sd(score))

    hex_cells %>% 
    right_join(hex_grid, .) %>% 
    filter(count >= 5) %>% 
    filter(TF %in% TFs_to_plot) %>% 
    group_by(TF) %>% 
    mutate(score = (score-mean(score))/sd(score)) %>% 
    ggplot() +
        geom_sf(aes(fill=score), color='grey') + 
        facet_wrap(~TF) +
        geom_vline(xintercept=log2(1+c(9,25)), linewidth=.2, color='darkgrey') + 
        # geom_jitter(data=points_df, aes(x=X, y=Y, color=score), width=.03, size=.3) +
    #     # stat_summary_hex(fun = function(x) if(length(x) > 10) {mean(x)} else {NA},
    #     #                  bins=9, alpha=.8) + 
        scale_color_paletteer_c("grDevices::Viridis", limits=c(-1,1), oob=squish) +
        scale_fill_paletteer_c("grDevices::Viridis", limits=c(-1,1), oob=squish) +
        scale_y_continuous(
            labels = function(x) x/5*60+40) +
        scale_x_continuous(
            name = 'Age',
            breaks = log2(1+c(0,1,3,5,9,25,60)),
            labels = function(x) round(2^x-1, 1)
        ) +
        guides(color='none') +
        theme_minimal() +
        theme(
            legend.position = 'right',
            strip.text.y = element_text(angle=0)
        )
}