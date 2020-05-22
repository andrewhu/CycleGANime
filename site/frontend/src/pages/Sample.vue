<template>
    <div id="samples">
        <h1 id="samples-title">Samples</h1>
        <p id="samples-description">
            Here's a short collection of the best results. This site only colorizes images in style of the top right
            square, but the pretrained models for the other styles as well as the code to run them are available in the
            <a href="https://github.com/andrewhu/CycleGANime" target="_blank">Github</a> repo.
        </p>
        <img src="/img/out1-min.jpg">
        <img src="/img/out2-min.jpg">
        <img src="/img/out3-min.jpg">
        <img src="/img/out4-min.jpg">
        <router-link to="/" id="samples-gohome">&larr; Back to home</router-link>
    </div>
</template>

<style>
    #samples {
        width: 80%;
        max-width: 800px;
        margin: 0 auto;
        padding-top: 10vh;
        padding-bottom: 100px;
    }

    #samples-title {
        margin-bottom: 10px;
        font-size: 2.4em;
    }

    #samples-description {
        font-family: "Proxima Nova", sans-serif;
        font-size: 1.2em;
        line-height: 1.2em;
        margin-bottom: 15px;
    }

    #samples img {
        margin: 0 auto;
        width: 100%;
        display: block;
        max-width: 512px;
        margin-bottom: 20px;
    }
    #samples-gohome {
        font-size: 1.4em;
        text-decoration: underline;
        cursor: pointer;
        transition: 0.2s ease;
    }
    #samples-gohome:hover {
        opacity: 0.6;
    }
</style>