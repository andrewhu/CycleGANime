<template>
    <div id="results">
        <h1 id="results-title">Results</h1>
        <h2 id="results-code">ID: <span>{{ this.$route.fullPath.slice(1).toUpperCase() }}</span></h2>
        <div id="results-wrap">
            <p id="results-processing-msg">Processing... this should take less than a minute.</p>
            <div id="loader">
                <span class="dot dot_1"></span>
                <span class="dot dot_2"></span>
                <span class="dot dot_3"></span>
                <span class="dot dot_4"></span>
            </div>
            <img class="result-img" id="pink">
<!--            <img class="result-img" id="ice">-->
            <p id="results-notice">For your privacy, results will be deleted from the server after 24 hours.</p>
        </div>

        <router-link to="/" id="results-gohome">&larr;Back to home</router-link>
    </div>
</template>

<script>
    import axios from 'axios';

    function show(elem, disp = 'block') {
        document.querySelector(elem).style.display = disp;
    }

    function hide(elem) {
        document.querySelector(elem).style.display = 'none';
    }

    export default {
        created() {
            var code = this.$route.fullPath.slice(1).toUpperCase()
            var this_ = this;

            function poll_result() {
                console.log("Polling result...")
                axios.get(`https://b2.drew.hu/file/peechee/cycleganime/results/${code}.jpg`, {responseType: "blob"})
                    .then(function (resp) {
                        if (resp.status === 200) {
                            // Show result
                            var reader = new window.FileReader();
                            reader.readAsDataURL(resp.data);
                            reader.onload = function () {
                                document.getElementById("pink").setAttribute("src", reader.result);
                                hide("#results-processing-msg");
                                hide("#loader")
                            }
                        } else {
                            // show error
                        }
                    }).catch(error => {
                    console.log("axios catch");
                    setTimeout(poll_result, 1000);
                })
            }

            poll_result()

            // axios.get(`/api/check/${code}`).then(function( resp) {
            //     if (resp.status === 200) {
            //         poll_result();
            //     }else {
            //         this_.$router.push({name: 'NotFound'});
            //     }
            // })



        }
    }


</script>

<style>
    #results {
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
        text-align: center;
        padding-top: 10vh;
        font-size: 2em;
    }
    @media(max-width: 768px) {
        #results {
            padding-top: 4vh;
        }
    }

    #results-title {
        font-size: 1.4em;
        margin-bottom: 15px;
    }

    #results-wrap {
        margin: 0 auto;
        margin-bottom: 20px;
        width: 80%;
    }

    #results-code {
        margin-bottom: 20px;
        font-family: "Ubuntu Mono", monospace;
    }

    #results-code span {
        border-radius: 4px;
        background: #fdbed3;
        padding: 4px;

        cursor: text;

    }

    #results-code span::selection {
        background: #d3fdef
    }

    #results-code span::-moz-selection {
        background: #d3fdef
    }
    .sampleContainer {
        margin-top: 250px;
    }

    #loader {
        position: relative;
        width: 44px;
        height: 8px;
        margin: 12px auto;
        padding-top: 20px;
    }

    .dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 4px;
        background: #f9276f;
        position: absolute;
    }

    .dot_1 {
        animation: animateDot1 1.5s linear infinite;
        left: 12px;
    }

    .dot_2 {
        animation: animateDot2 1.5s linear infinite;
        animation-delay: 0.5s;
        left: 24px;
    }

    .dot_3 {
        animation: animateDot3 1.5s linear infinite;
        left: 12px;
    }

    .dot_4 {
        animation: animateDot4 1.5s linear infinite;
        animation-delay: 0.5s;
        left: 24px;
    }

    @keyframes animateDot1 {
        0% {
            transform: rotate(0deg) translateX(-12px);
        }
        25% {
            transform: rotate(180deg) translateX(-12px);
        }
        75% {
            transform: rotate(180deg) translateX(-12px);
        }
        100% {
            transform: rotate(360deg) translateX(-12px);
        }
    }
    @keyframes animateDot2 {
        0% {
            transform: rotate(0deg) translateX(-12px);
        }
        25% {
            transform: rotate(-180deg) translateX(-12px);
        }
        75% {
            transform: rotate(-180deg) translateX(-12px);
        }
        100% {
            transform: rotate(-360deg) translateX(-12px);
        }
    }
    @keyframes animateDot3 {
        0% {
            transform: rotate(0deg) translateX(12px);
        }
        25% {
            transform: rotate(180deg) translateX(12px);
        }
        75% {
            transform: rotate(180deg) translateX(12px);
        }
        100% {
            transform: rotate(360deg) translateX(12px);
        }
    }
    @keyframes animateDot4 {
        0% {
            transform: rotate(0deg) translateX(12px);
        }
        25% {
            transform: rotate(-180deg) translateX(12px);
        }
        75% {
            transform: rotate(-180deg) translateX(12px);
        }
        100% {
            transform: rotate(-360deg) translateX(12px);
        }
    }

    .result-img {
        width: 100%;
        max-width: 416px;
        margin-bottom: 5px;
    }
    #results-notice {
        font-size: 0.7em;
    }
    #results-gohome {
        font-size: 0.8em;
    }

</style>